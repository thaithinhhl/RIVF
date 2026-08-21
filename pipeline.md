# Pipeline kỹ thuật của hệ thống

Tài liệu này mô tả **pipeline đang được dùng cho kết quả chính**, không mô tả phương án adaptive semantic–graph fusion cũ như thể đó là main method.

- Minh hoạ trực quan: [`pipeline.html`](pipeline.html)
- Kế hoạch thí nghiệm: [`experiment.md`](experiment.md)
- Kết quả theo Research Question: [`version1.md`](version1.md)

## 1. Mục tiêu

Input của hệ thống là một câu hỏi multi-hop QA và tập passage đi kèm. Hệ thống phải chọn một context nhỏ, tối đa **200 evidence tokens**, nhưng vẫn chứa đủ chuỗi bằng chứng để LLM trả lời.

```text
Question + passages
    → sentence graph
    → seed retrieval
    → graph expansion ≤2-hop
    → semantic và structural signals
    → label-free routing
    → conditional evidence selection
    → pair-aware pruning ≤200 tokens
    → LLM answer
```

## 2. Phạm vi corpus và đơn vị node

Trong thiết lập HotpotQA distractor và 2Wiki hiện tại, mỗi `QAItem` đã chứa một tập passage ứng viên. Graph được xây **riêng cho từng câu hỏi** trên tập passage này; pipeline chưa truy vấn một index toàn bộ Wikipedia.

```text
Dataset
└── QAItem
    ├── question
    ├── passages
    │   ├── Passage A
    │   │   ├── Sentence A0
    │   │   └── Sentence A1
    │   └── Passage B
    │       ├── Sentence B0
    │       └── Sentence B1
    └── gold answer/supporting facts (chỉ dùng khi đánh giá)
```

Passage là nguồn dữ liệu; **mỗi sentence mới là một graph node**:

```python
Sentence(
    passage_title="Inception",
    sent_id=0,
    text="Inception was directed by Christopher Nolan."
)
```

Node key là `(passage_title, sent_id)`. Khi embedding/reranking, sentence được biểu diễn dưới dạng:

```text
Inception: Inception was directed by Christopher Nolan.
```

Title prefix giúp các câu chứa đại từ hoặc tham chiếu ngầm vẫn có ngữ cảnh entity. Khi đếm evidence tokens, hệ thống chỉ đếm `sentence.text`; title và instruction được tính riêng trong `prompt_tokens`.

## 3. Bước 1 — Seed retrieval

### Input

- Câu hỏi `q`.
- Toàn bộ sentence nodes trong passage set của câu hỏi.

### Xử lý

BAAI/bge-m3 tạo embedding cho câu hỏi và `title: sentence`. Hệ thống tính cosine similarity:

```text
SeedScore(q, v) = cosine(Embed(q), Embed(title(v) + text(v)))
```

Sau đó lấy Top-5 sentence nodes theo mặc định:

```text
S0 = {s1, s2, s3, s4, s5}
```

### Output

`S0` là seed set dùng làm nguồn cho multi-source graph traversal. Seed retrieval luôn dùng bi-encoder vì bước này phải chấm toàn bộ sentence set.

## 4. Bước 2 — Xây graph

Graph vô hướng được xây mới cho từng câu hỏi.

### Nodes

```text
V = tất cả sentence trong các passage của QAItem
```

### Edges

| Edge | Điều kiện nối | Vai trò |
|---|---|---|
| `same_passage` | Hai sentence liền nhau trong cùng passage | Giữ mạch nội dung cục bộ |
| `entity_overlap` | Hai sentence khác passage có entity giao nhau | Nối evidence qua entity chung |
| `title_mention` | Sentence ở passage A nhắc title của passage B | Tạo bridge giữa hai bài viết |

`title_mention` có edge weight 2; hai edge còn lại có weight 1. BFS expansion đếm hop không trọng số, còn PPR có sử dụng edge weight.

Gold answer và supporting facts không được dùng để tạo node, edge hoặc seed.

## 5. Bước 3 — Graph expansion

Hệ thống chạy multi-source BFS từ năm seed nodes và giữ node có khoảng cách tối đa hai cạnh:

```text
C = {v ∈ V | distance(v, S0) ≤ 2}
```

Candidate pool gồm:

- Các seed, hop distance 0.
- Các node 1-hop.
- Các node 2-hop.

Mỗi candidate lưu:

- `hop_distance`.
- Một shortest-path parent/path từ seed.
- Các graph neighbors nằm trong candidate pool.
- `semantic_score`.
- `graph_score`.
- Conditional score/anchor nếu có.

Graph expansion là **candidate generator**: node ngoài phạm vi hai hop không được rerank hoặc đưa vào context.

## 6. Bước 4 — Tính Semantic và GraphRel

Sau expansion, hệ thống vẫn tính cả hai tín hiệu cho mọi candidate.

### 6.1 Semantic score

Ở main pipeline, cross-encoder BAAI/bge-reranker-v2-m3 chấm trực tiếp từng cặp:

```text
Semantic(q, v) = CrossEncoder(q, title(v) + text(v))
```

Cross-encoder chỉ chạy trên candidate pool đã được graph giới hạn, không quét toàn bộ corpus.

### 6.2 Graph score

Code hỗ trợ hai cách:

```text
HopGraphRel(v) = 1 / (1 + hop_distance(v, S0))
```

hoặc Personalized PageRank:

```text
PPR(v) = PageRank mass với restart distribution đặt đều trên seeds
GraphRel(v) = PPR(v) / max_{u∈C} PPR(u)
```

Main configs hiện đặt `graph_rel_method: ppr`, nên `graph_score` vẫn được tính và lưu.

### 6.3 Graph score có tham gia main ranking không?

Main configs đặt:

```yaml
alpha: 1.0
adaptive_alpha: false
```

Do đó base relevance là:

```text
Base(v) = α·Semantic(v) + (1−α)·GraphRel(v)
        = Semantic(v)
```

Graph vẫn được sử dụng để tạo candidate pool, xác định neighbor/path và xây anchor–endpoint pair. Chỉ có giá trị `graph_score` không được cộng trực tiếp vào main ranking.

Lý do: controlled alpha sweep cho thấy semantic-only tốt hơn linear blend trên cả Hotpot và 2Wiki; graph proximity/PPR thường tăng điểm cho seed hoặc node trung tâm nhưng không chắc liên quan tới câu hỏi.

## 7. Bước 5 — Label-free question router

Router chỉ đọc text câu hỏi, không sử dụng trường `item.type` hoặc gold label. Nó chọn một trong ba route:

| Route | Ý nghĩa | Selection strategy |
|---|---|---|
| `direct_comparison` | So sánh trực tiếp các entity/thuộc tính nêu trong câu hỏi | `coverage_topk` |
| `bridge_comparison` | Mỗi phía cần tìm intermediate relation rồi mới so sánh | Single-chain mặc định; dual-chain là mode tùy chọn |
| `chained_reasoning` | Câu hỏi cần theo chuỗi quan hệ qua bridge entity | `conditional_topk` |

Router là heuristic label-free dựa trên pattern của câu hỏi. Việc đánh giá theo dataset question type chỉ xảy ra sau inference.

## 8. Bước 6 — Conditional second-hop scoring

Đây là tín hiệu thay thế an toàn hơn cho việc cộng PPR trực tiếp vào relevance.

### 8.1 Chọn anchor

Trong QA mode chính, hệ thống lấy một candidate semantic mạnh từ một passage làm anchor:

```text
a* = argmax Semantic(q, a)
```

`conditional_anchor_n=1` trong main configs.

### 8.2 Tìm endpoint liên kết

Chỉ các graph neighbors khác passage với anchor mới được chấm conditional:

```text
N_cross(a*) = {v | (a*,v) ∈ E và title(v) ≠ title(a*)}
```

### 8.3 Conditional cross-encoder

```text
Conditional(q, a*, v)
  = CrossEncoder(
      question + "Known evidence:" + text(a*),
      title(v) + text(v)
    )
```

Điều này giúp một answer-bearing endpoint có semantic similarity thấp với câu hỏi gốc vẫn được cứu khi nó hợp lý sau anchor.

Mỗi endpoint lưu `conditional_score` và `conditional_anchor_key` tốt nhất.

## 9. Bước 7 — Route-specific pruning

### 9.1 Direct comparison

`coverage_topk` percentile-normalize base relevance rồi greedily chọn candidate vừa budget. Main configs hiện không đặt `coverage_bonus`, nên giá trị mặc định bằng 0: route này thực chất là **semantic budget-fill có skip-over**, chưa có bonus đa dạng passage.

### 9.2 Chained reasoning và bridge QA mode

`conditional_topk` tính:

```text
FinalRank(v)
  = percentile(Base(v))
  + 0.30 × percentile(Conditional(v))
```

Nếu một endpoint có conditional anchor, selector cố đưa cả cặp vào context:

```text
pair(v) = {anchor(v), v}
```

Nếu cả cặp không vừa budget, cặp đó bị bỏ thay vì giữ một endpoint cô lập.

### 9.3 Dual-chain evidence-recall mode

Với `bridge_policy: dual_chain`, `chain_set_topk` cố dành budget cho một pair của mỗi anchor trước khi lấp phần còn lại. Chế độ này tăng 2Wiki SP Recall nhưng giảm Answer F1, nên không dùng làm headline QA method.

### 9.4 Hard token ceiling

Token được đếm bằng tokenizer `cl100k_base`:

```text
Σ TokenCount(sentence.text) ≤ 200
```

Selector bỏ qua candidate/pair không vừa phần budget còn lại và tiếp tục xét candidate sau nếu strategy hỗ trợ budget filling.

## 10. Bước 8 — Context và prompt

Main routed configs hiện không bật `chain_aware_prompt`. Selected sentences được group lại theo passage title và thứ tự sentence gốc:

```text
[Passage title A]
selected sentence A0
selected sentence A2

[Passage title B]
selected sentence B1
```

Chain-aware serialization theo selection order tồn tại cho ablation và có thể thêm reasoning hint theo route, nhưng không thuộc main result hiện tại.

Prompt yêu cầu:

- Chỉ dùng context được cung cấp.
- Yes/no trả đúng `yes` hoặc `no`.
- Câu hỏi khác trả lời bằng exact name/phrase ngắn nhất.

## 11. Bước 9 — Answer generation

```text
Answer = gpt-4o-mini(prompt, temperature=0)
```

Cùng answer model và prompt policy phải được dùng cho mọi method trong cùng experiment. Thay đổi retrieval method không được đồng thời thay đổi LLM hoặc instruction.

## 12. Bước 10 — Logging và evaluation

Mỗi result row cần ghi:

- Dataset, QID, method và seed.
- Budget, alpha và computed route.
- Selected sentence keys/text.
- SP Recall, SP Precision và SP F1.
- Answer EM và Answer F1.
- Evidence tokens và số nodes.
- Prompt tokens cho các run mới.
- Retrieval/selection latency.

Ba token quantities không được đánh tráo:

| Đại lượng | Bao gồm |
|---|---|
| `evidence_tokens` / `context_tokens` | Text của selected evidence |
| `prompt_tokens` | Instruction + title/format + evidence + question |
| `output_tokens` | Câu trả lời do LLM sinh |

Pain point chính được đo bằng evidence/context tokens; RQ6 phải báo thêm prompt/output tokens và chi phí end-to-end.

## 13. Pipeline của năm methods trong Main Comparison

| Method | Candidate/ranking pipeline | Graph | Pruning | Budget |
|---|---|---|---|---:|
| Dense BI | Bi-encoder score trên mọi sentence | Không | Semantic top/budget | 600 |
| Dense CE | Cross-encoder score trên mọi sentence | Không | Semantic top/budget | 600 |
| Graph 1-hop CE | Seeds → expansion 1-hop → CE | 1-hop | Semantic top/budget | 600 |
| Graph 2-hop CE | Seeds → expansion 2-hop → CE | 2-hop | Semantic top/budget | 600 |
| **Ours** | Seeds → 2-hop → CE → router → conditional scoring | 2-hop structure | Route-specific, pair-aware | **200** |

Dense CE là controlled semantic baseline. Graph 1/2-hop đo lợi ích của expansion khi chưa có routing và pair-aware conditional selection.

## 14. Main pipeline và adaptive fusion cũ

### Main pipeline hiện tại

```text
C
├── Semantic(q,v)
├── Graph structure: path/neighbors
└── Router
      ├── direct → semantic budget-fill
      └── chained → Conditional(q,anchor,v) → pair-aware pruning
```

### Adaptive fusion cũ/ablation

```text
C
├── Top-K Semantic → S_sem
├── Top-K GraphRel → S_graph
└── disagreement + confidence + seed miss
        → α(q)
        → α(q)·Semantic + (1−α(q))·GraphRel
```

Code adaptive vẫn tồn tại để chạy RQ4, nhưng main configs không bật `adaptive_alpha`. Không nên mô tả adaptive fusion là contribution chính khi viết kết quả hiện tại.

Một biến thể graph-gated rescue cũng đã được thử: query-weighted PPR chỉ cộng residual cho `S_graph − S_sem` khi conditional CE xác nhận endpoint. Trên n=200, biến thể này không cải thiện Hotpot và làm 2Wiki Answer F1 giảm `0.5940 → 0.5865`; do đó strategy vẫn chỉ là ablation, không thuộc main pipeline.

## 15. Pseudocode main method

```text
INPUT: question q, passages P, budget B=200

V ← split_passages_into_sentence_nodes(P)
G ← build_graph(V)
S0 ← top_5_by_bge_m3(q, V)
C ← bfs_expand(G, S0, max_hops=2)

for v in C:
    sem[v]   ← CrossEncoder(q, v)
    graph[v] ← normalized_PPR(G, S0, v)  # logged/ablation signal

route ← label_free_router(q)

if route == direct_comparison:
    selected ← semantic_budget_fill(C, B)
else:
    anchor ← strongest_semantic_anchor(C)
    for v in cross_passage_neighbors(anchor):
        cond[v] ← CrossEncoder(q + anchor.text, v.text)
    selected ← pair_aware_select(C, sem, cond, weight=0.30, budget=B)

context ← group_selected_by_original_passage_order(selected)
answer  ← LLM(q, context, temperature=0)

OUTPUT: answer, selected evidence, metrics, token/latency logs
```

## 16. Code và config tương ứng

| Chức năng | File |
|---|---|
| Data model/title prefix | `src/rivf/data/schema.py` |
| Seed retrieval | `src/rivf/retrieval/seed.py` |
| Graph construction/BFS/PPR | `src/rivf/graph/build.py` |
| Candidate và conditional scoring | `src/rivf/retrieval/scoring.py` |
| Label-free router | `src/rivf/retrieval/question_router.py` |
| Route-specific pruning | `src/rivf/retrieval/pruning.py` |
| Prompt serialization | `src/rivf/generation/prompts.py` |
| Experiment orchestration | `src/rivf/experiments/run_experiment.py` |
| Adaptive fusion ablation | `src/rivf/retrieval/adaptive_alpha.py` |
| Hotpot full config | `configs/final_hotpot_full_retrieval.yaml` |
| 2Wiki full config | `configs/final_2wiki_full_retrieval.yaml` |

## 17. Các invariant cần giữ khi phát triển

1. Không sử dụng answer, supporting facts hoặc dataset type trong inference pipeline.
2. Ours không được vượt hard ceiling 200 evidence tokens trong main experiment.
3. Baseline và Ours phải dùng cùng QID, prompt, LLM và temperature.
4. Graph expansion tối đa 2-hop trừ khi ablation ghi rõ khác biệt.
5. Không gọi `graph_score` là main ranking signal khi `alpha=1`.
6. Không gọi selected-evidence tokens là total prompt tokens.
7. Mọi thay đổi pipeline phải được kiểm chứng trên cả Hotpot và 2Wiki.
