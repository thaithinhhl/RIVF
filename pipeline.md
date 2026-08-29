# GRAFT-v1 — Canonical Technical Pipeline

Tài liệu này là đặc tả duy nhất của main method. Các biến thể PPR,
semantic–graph fusion và graph-rescue trong config cũ chỉ là lịch sử/ablation,
không phải GRAFT-v1.

## 1. Bài toán và phạm vi

Input là câu hỏi `q` và candidate passages `P` do benchmark cung cấp. Hệ thống
không truy vấn toàn bộ Wikipedia. Vì vậy task được báo cáo là **multi-hop
evidence selection trên provided candidate passages**, sau đó sinh câu trả lời.

Main benchmarks:

- HotpotQA dev-distractor.
- 2WikiMultiHopQA dev.

Raw data không bị sửa. Trước khi chạy, loader gộp passage trùng hoàn toàn và
loại QID có supporting-fact reference không tồn tại để mọi method có cùng
metric ceiling và cùng candidate identities.

## 2. Ba stage của GRAFT

```text
Question + candidate passages
    ↓
1. Graph-Guided Chain Proposal
   sentence graph → semantic seeds → expansion ≤H hops
    ↓
2. Query-Conditioned Chain Verification
   node CE → anchors → conditional link CE → validated links
    ↓
3. Chain-Preserving Evidence Compression
   singleton/pair utilities → atomic budgeted selection
    ↓
Compressed evidence → GPT-4o mini → answer
```

Nguyên lý trung tâm:

```text
graph proposes → semantics verifies → budgeted selection preserves
```

Graph connectivity không được coi là relevance. Graph chỉ đề xuất candidate và
quan hệ cấu trúc; cross-encoder quyết định query-conditioned relevance.

## 3. Hyperparameter được freeze

| Ký hiệu | Ý nghĩa | Main value |
|---|---|---:|
| `K` | Số semantic seeds | 5 |
| `H` | Số hop expansion tối đa | 2 |
| `M` | Số semantic anchors | 2 |
| `L` | Validated links tối đa cho mỗi anchor | 2 |
| `β` | Trọng số conditional link score | 0.30 |
| `B` | Hard evidence-token ceiling | 200 |

Các giá trị này giống nhau trên HotpotQA và 2Wiki. Mọi tuning/sensitivity phải
dùng development manifest, không dùng evaluation manifest.

## 4. Stage 1 — Graph-Guided Chain Proposal

### 4.1 Sentence nodes

Mỗi sentence là một node:

```text
v = (passage_title, sent_id, text)
```

Biểu diễn đưa vào embedding/reranker là `title: sentence`. Evidence budget chỉ
đếm `sentence.text`; prompt tokens được log riêng.

### 4.2 Evidence graph

Xây graph vô hướng riêng cho từng câu hỏi:

```text
G = (V, E_G)
```

Ba edge type:

| Edge | Điều kiện |
|---|---|
| `same_passage` | Hai sentence liền nhau trong cùng passage |
| `entity_overlap` | Hai sentence khác passage có named entity chung |
| `title_mention` | Sentence ở passage A nhắc title của passage B |

Main dùng cả ba edge. Code cho phép bật/tắt từng edge để chạy ablation.

### 4.3 Seed retrieval và expansion

BAAI/bge-m3 tính:

```text
s_seed(q,v) = cosine(Embed(q), Embed(title(v) + text(v)))
S0 = TopK_v s_seed(q,v)
```

Sau đó multi-source BFS tạo candidate pool:

```text
C = {v ∈ V | distance(v,S0) ≤ H}
```

Main dùng `K=5`, `H=2`. Gold answer, supporting facts và benchmark type không
được dùng trong graph construction, seed retrieval hoặc expansion.

## 5. Stage 2 — Query-Conditioned Chain Verification

### 5.1 Direct node relevance

BAAI/bge-reranker-v2-m3 chấm mọi candidate:

```text
s_node(q,v) = CE(q, title(v) + text(v))
```

### 5.2 Label-free routing

Router chỉ đọc question text và không đọc `QAItem.type`:

- `direct_comparison`: có comparison cue nhưng không có intermediate-relation cue.
- `bridge_comparison`: có cả comparison cue và intermediate-relation cue.
- `chained_reasoning`: các trường hợp còn lại.

Các cue regex được khai báo tường minh trong
`src/rivf/retrieval/question_router.py`. Direct route không chạy conditional CE;
hai route còn lại chạy chain verification.

### 5.3 Anchor selection

Với route cần chain, lấy tối đa `M=2` anchor từ các passage khác nhau:

```text
A = TopM_a s_node(q,a)
```

### 5.4 Conditional link scoring

Với mỗi anchor `a`, chỉ chấm cross-passage graph neighbors:

```text
N_cross(a) = {v | (a,v) ∈ E_G and title(v) != title(a)}
s_link(q,a,v) = CE(q ⊕ "Known evidence:" ⊕ a, v)
```

### 5.5 Định nghĩa validated link

Một link không được gọi là validated chỉ vì có graph edge. Main định nghĩa:

```text
L_a = TopL_{v ∈ N_cross(a)} s_link(q,a,v),  L=2
L_valid = union_a {(a,v) | v ∈ L_a}
```

Code hỗ trợ thêm `conditional_link_min_score=τ`; main để `τ=None` vì raw CE
logit không được giả định calibrated giống nhau giữa dataset. Nếu một endpoint
được nhiều anchor validate, giữ anchor có conditional score cao nhất.

## 6. Stage 3 — Chain-Preserving Evidence Compression

Score được percentile-normalize trong từng câu hỏi:

```text
node(v) = percentile(s_node(q,v))
link(a,v) = percentile(s_link(q,a,v))
```

Evidence unit có hai dạng:

```text
singleton: u = {v}
validated pair: u = {a,v}, với (a,v) ∈ L_valid
```

Utility:

```text
U(q,{v}) = node(v)

U(q,{a,v}) = node(a) + node(v) + β·link(a,v)
```

Main dùng `β=0.30`. Selector xếp unit theo utility và greedily nhận unit vừa
budget:

```text
sum TokenCount(sentence.text) ≤ B,  B=200
```

Pair là atomic: nếu toàn bộ `{a,v}` không vừa budget, bỏ cả pair; không giữ
endpoint `v` cô lập. Nếu anchor đã được chọn bởi unit trước, pair chỉ trả thêm
chi phí của endpoint nhưng vẫn bảo đảm context cuối chứa cả hai node.

Direct route dùng semantic budget-fill vì không có validated chain unit.

## 7. Context và generation

Selected sentences được group theo passage title và sentence order gốc. Mọi
method trong cùng experiment dùng cùng prompt và:

```text
gpt-4o-mini-2024-07-18, temperature=0
```

Result row log cả model alias được yêu cầu và model thực tế API trả về, response
ID, system fingerprint và API token usage khi các trường này có sẵn.

## 8. Complexity

Không tính chi phí encode embeddings đã cache:

```text
Seed retrieval:          O(|V|)
Graph traversal:         O(|V| + |E_G|)
Node cross-encoding:     O(|C|)
Conditional scoring:     O(sum_{a∈A} |N_cross(a)|)
Unit selection:          O(|C| log |C|)
```

## 9. Algorithm 1

```text
INPUT: question q, candidate passages P
PARAMETERS: K=5, H=2, M=2, L=2, beta=0.30, B=200

V  ← split_into_sentence_nodes(P)
G  ← build_graph(V, adjacency + entity_overlap + title_mention)
S0 ← top_K_by_bge_m3(q, V)
C  ← bfs_expand(G, S0, max_hops=H)

for v in C:
    node[v] ← CE(q, v)

route ← label_free_router(q)

if route == direct_comparison:
    units ← {{v}: v in C}
else:
    A ← top_M_distinct_passage_anchors(C, node)
    for a in A:
        scores ← {v: CE(q + a, v) for v in cross_passage_neighbors(a)}
        validated[a] ← top_L(scores)
    units ← singleton_units(C) union validated_pair_units(validated)

for unit u:
    utility[u] ← node utility, plus beta * link utility for pairs

E ← greedy_atomic_select(units, budget=B)
answer ← LLM(q, E, temperature=0)

OUTPUT: answer, selected evidence, routes, validated links, metrics, provenance
```

## 10. Code mapping

| Block | File |
|---|---|
| Data hygiene/loading | `src/rivf/data/hotpotqa.py` |
| Graph/edge ablation | `src/rivf/graph/build.py` |
| Seeds | `src/rivf/retrieval/seed.py` |
| Node/link verification | `src/rivf/retrieval/scoring.py` |
| Router | `src/rivf/retrieval/question_router.py` |
| Atomic pair selection | `src/rivf/retrieval/pruning.py` |
| Prompt/LLM | `src/rivf/generation/` |
| Runner/provenance | `src/rivf/experiments/run_experiment.py` |
| Canonical configs | `configs/README.md` |

## 11. Invariants

1. Không dùng answer, supporting facts hoặc dataset type trong inference.
2. Main GRAFT không cộng graph proximity/PPR vào relevance.
3. Chỉ Top-L conditional links mới được gọi là validated.
4. Validated pair được giữ hoặc bỏ nguyên khối.
5. Main GRAFT không vượt 200 evidence tokens.
6. Mọi method dùng cùng canonicalized candidate passages và QIDs.
7. Development và evaluation manifests không giao nhau.
8. Pilot/PPR/graph-rescue results cũ không được báo cáo như GRAFT-v1 final.
