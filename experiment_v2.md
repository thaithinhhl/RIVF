# Experimental Design v2 — Ours sử dụng Graph Score

## 1. Mục tiêu

Đánh giá liệu **Ours v2**, khi sử dụng graph score, có thể giữ chất lượng multi-hop QA cạnh tranh với các baseline dùng tối đa 600 evidence tokens trong khi Ours bị giới hạn ở 200 tokens hay không.

HotpotQA và 2WikiMultiHopQA được đánh giá bằng **cùng pipeline, cùng hyperparameter và cùng protocol**. Kết quả của cả hai benchmark được trình bày trong một bảng chính thống nhất.

## 2. Ours v2 được freeze trước khi chạy

Ours v2 sử dụng graph score theo cơ chế gated rescue, không dùng linear blend trên toàn bộ candidate pool.

```text
Passages → sentence nodes
→ seed retrieval Top-5
→ graph expansion ≤2-hop
→ Semantic(q,v) + query-weighted PPR(v)
→ S_sem và S_graph
→ semantic–graph disagreement gate
→ conditional CE(q + anchor, endpoint)
→ graph-gated pair score
→ pair-aware pruning ≤200 tokens
→ LLM answer
```

### 2.1 Các score

```text
Semantic(v) = CE(q, v)

SeedWeight(s) = softmax(Semantic(q,s) / τ)

GraphRel(v) = WeightedPPR(v) / max WeightedPPR

Conditional(v) = CE(q + anchor(v), v)
```

### 2.2 Graph-gated rescue

```text
S_sem   = Top-6 theo Semantic
S_graph = Top-6 theo GraphRel

disagreement(q)
  = 1 - |S_sem ∩ S_graph| / |S_sem ∪ S_graph|

λ(q) = λ_max × disagreement(q)
```

Chỉ candidate thỏa cả ba điều kiện sau mới nhận graph residual:

1. `v ∈ S_graph − S_sem`.
2. `v` nối với một semantic anchor trong graph.
3. `Conditional(v)` tồn tại và xác nhận endpoint.

```text
Rescue(v)
  = GraphRank(v) × ConditionalRank(v) × HopPenalty(v)

FinalRank(v)
  = SemanticRank(v)
  + 0.30 × ConditionalRank(v)
  + λ(q) × Rescue(v)
```

### 2.3 Hyperparameter dùng chung

| Parameter | Giá trị freeze |
|---|---:|
| Seed nodes | 5 |
| Maximum hops | 2 |
| Graph score | Query-weighted PPR |
| PPR seed temperature `τ` | 0.20 |
| Semantic/graph proxy K | 6 |
| Conditional weight | 0.30 |
| Maximum graph weight `λ_max` | 0.20 |
| Ours evidence budget | 200 |
| QA policy | Routed single-chain |

Không điều chỉnh `λ_max` riêng cho Hotpot và 2Wiki sau khi nhìn kết quả final.

## 3. Research Questions

| RQ | Câu hỏi | Thí nghiệm trả lời |
|---|---|---|
| **RQ1** | Ours v2-200 có cạnh tranh với baseline-600 trên cả hai benchmark? | Unified Main Comparison |
| **RQ2** | Graph score có thực sự đóng góp so với cùng pipeline không graph score? | Graph Contribution Ablation |
| **RQ3** | Lợi thế có ổn định theo token budget? | Budget Sweep/Pareto Analysis |
| **RQ4** | Thành phần nào của graph-gated rescue tạo ra cải thiện? | Component Ablation |
| **RQ5** | Graph score giúp/hại loại câu hỏi nào? | Type-wise Robustness/Error Analysis |
| **RQ6** | Ours giảm bao nhiêu context, latency và chi phí? | Efficiency/Cost Analysis |

## 4. Methods trong bảng chính

| Method | Mô tả | Budget |
|---|---|---:|
| Dense CE | Cross-encoder ranking, không graph | 600 |
| Graph 1-hop CE | Graph expansion 1-hop + CE ranking | 600 |
| Graph 2-hop CE | Graph expansion 2-hop + CE ranking | 600 |
| **Ours v2 Graph-Gated** | 2-hop + CE + weighted PPR + router + conditional pair rescue | **200** |

Ours dùng cross-encoder, vì vậy cả ba baseline chính cũng dùng đúng cross-encoder đó. Bi-encoder chỉ còn ở bước seed retrieval chung và không được báo cáo như một baseline riêng.

Mọi method sử dụng cùng passage input, QID, semantic reranker BAAI/bge-reranker-v2-m3, answer model, prompt template và tokenizer. Các graph-based methods dùng chung BAAI/bge-m3 để lấy seed; Dense CE không cần seed vì chấm trực tiếp toàn bộ sentence set. Khác biệt được kiểm soát nằm ở graph expansion, graph score, routing và pruning policy.

## 5. RQ1 — Unified Main Comparison

### 5.1 Bảng chính dùng trong paper

Điền mean trên toàn bộ evaluation set. Bold giá trị tốt nhất; underline giá trị tốt thứ hai. Không bold token cao.

| Dataset | Method | Budget | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Evidence Tokens ↓ | Nodes ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | Dense CE | 600 | — | — | — | — | — | — | — |
| HotpotQA | Graph 1-hop CE | 600 | — | — | — | — | — | — | — |
| HotpotQA | Graph 2-hop CE | 600 | — | — | — | — | — | — | — |
| HotpotQA | **Ours v2 Graph-Gated** | **200** | — | — | — | — | — | — | — |
| 2Wiki | Dense CE | 600 | — | — | — | — | — | — | — |
| 2Wiki | Graph 1-hop CE | 600 | — | — | — | — | — | — | — |
| 2Wiki | Graph 2-hop CE | 600 | — | — | — | — | — | — | — |
| 2Wiki | **Ours v2 Graph-Gated** | **200** | — | — | — | — | — | — | — |

### 5.2 Bảng paired compression và non-inferiority

Mỗi Ours row được ghép QID với comparator tương ứng.

| Dataset | Comparator | Token ratio ↑ | ΔSP Recall | ΔSP F1 | ΔAnswer EM | ΔAnswer F1 | 95% CI của ΔAnswer F1 | Kết luận |
|---|---|---:|---:|---:|---:|---:|---|---|
| HotpotQA | Dense CE | — | — | — | — | — | — | — |
| HotpotQA | Graph 1-hop CE | — | — | — | — | — | — | — |
| HotpotQA | Graph 2-hop CE | — | — | — | — | — | — | — |
| 2Wiki | Dense CE | — | — | — | — | — | — | — |
| 2Wiki | Graph 1-hop CE | — | — | — | — | — | — | — |
| 2Wiki | Graph 2-hop CE | — | — | — | — | — | — | — |

```text
Token ratio = comparator mean evidence tokens / Ours mean evidence tokens
Delta       = Ours metric - comparator metric
```

Answer F1 được xem là non-inferior khi lower bound của paired bootstrap CI không thấp hơn `−0.02`. Ngưỡng phải được công bố trước khi chạy full result.

## 6. RQ2 — Graph Contribution Ablation

Đây là thí nghiệm bắt buộc vì main method đã quyết định sử dụng graph score.

| Dataset | Variant | SP Recall | SP Precision | SP F1 | Answer EM | Answer F1 | Tokens | ΔAnswer F1 vs Full |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | **Full Ours v2** | — | — | — | — | — | — | — |
| HotpotQA | `−GraphScore` (`λ_max=0`) | — | — | — | — | — | — | — |
| HotpotQA | Linear blend Semantic+PPR | — | — | — | — | — | — | — |
| 2Wiki | **Full Ours v2** | — | — | — | — | — | — | — |
| 2Wiki | `−GraphScore` (`λ_max=0`) | — | — | — | — | — | — | — |
| 2Wiki | Linear blend Semantic+PPR | — | — | — | — | — | — | — |

RQ2 chỉ được trả lời “graph score có ích” khi Full Ours cải thiện ít nhất một metric evidence và Answer F1 không vi phạm non-inferiority margin.

## 7. RQ3 — Budget Sweep và Pareto Frontier

Budget: `100, 150, 200, 300, 400, 600`.

| Dataset | Method | 100 | 150 | 200 | 300 | 400 | 600 |
|---|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | Dense CE — Answer F1 | — | — | — | — | — | — |
| HotpotQA | Graph 2-hop CE — Answer F1 | — | — | — | — | — | — |
| HotpotQA | Ours v2 — Answer F1 | — | — | — | — | — | — |
| 2Wiki | Dense CE — Answer F1 | — | — | — | — | — | — |
| 2Wiki | Graph 2-hop CE — Answer F1 | — | — | — | — | — | — |
| 2Wiki | Ours v2 — Answer F1 | — | — | — | — | — | — |

Tạo hai figure từ cùng output:

1. Answer F1 theo evidence tokens.
2. SP F1 theo evidence tokens.

## 8. RQ4 — Component Ablation

| Variant | Thay đổi so với Full Ours v2 | Mục đích |
|---|---|---|
| Full Ours v2 | Không thay đổi | Reference |
| Uniform PPR | Bỏ semantic seed weighting | Kiểm tra query-weighted restart |
| `−DisagreementGate` | Dùng λ cố định | Kiểm tra query-level gate |
| `−ConditionalValidation` | Graph candidate không cần CE xác nhận | Kiểm tra chống graph distractor |
| `−PairPreservation` | Chọn endpoint độc lập | Kiểm tra chain completeness |
| `−Router` | Mọi câu dùng cùng strategy | Kiểm tra label-free routing |
| Graph-only | Bỏ semantic base relevance | Sanity lower bound |

Template kết quả:

| Dataset | Variant | SP Recall | SP F1 | Answer EM | Answer F1 | Tokens |
|---|---|---:|---:|---:|---:|---:|
| HotpotQA | Full Ours v2 | — | — | — | — | — |
| HotpotQA | Uniform PPR | — | — | — | — | — |
| HotpotQA | −DisagreementGate | — | — | — | — | — |
| HotpotQA | −ConditionalValidation | — | — | — | — | — |
| HotpotQA | −PairPreservation | — | — | — | — | — |
| HotpotQA | −Router | — | — | — | — | — |
| 2Wiki | Full Ours v2 | — | — | — | — | — |
| 2Wiki | Uniform PPR | — | — | — | — | — |
| 2Wiki | −DisagreementGate | — | — | — | — | — |
| 2Wiki | −ConditionalValidation | — | — | — | — | — |
| 2Wiki | −PairPreservation | — | — | — | — | — |
| 2Wiki | −Router | — | — | — | — | — |

## 9. RQ5 — Type-wise Robustness và Error Analysis

### 9.1 HotpotQA

| Question type | N | SP Recall | SP F1 | Answer EM | Answer F1 | Tokens | Graph rescue rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bridge | — | — | — | — | — | — | — |
| Comparison | — | — | — | — | — | — | — |

### 9.2 2WikiMultiHopQA

| Question type | N | SP Recall | SP F1 | Answer EM | Answer F1 | Tokens | Graph rescue rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Comparison | — | — | — | — | — | — | — |
| Inference | — | — | — | — | — | — | — |
| Compositional | — | — | — | — | — | — | — |
| Bridge comparison | — | — | — | — | — | — | — |

Mỗi failure được gán một nguyên nhân: seed miss, graph reachability miss, router error, graph false rescue, budget overflow, pair pruning error hoặc generation error.

## 10. RQ6 — Efficiency và Cost

| Dataset | Method | Evidence tokens | Prompt tokens | Output tokens | Nodes | Retrieval latency | LLM latency | Estimated cost/question |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | Dense CE | — | — | — | — | — | — | — |
| HotpotQA | Graph 1-hop CE | — | — | — | — | — | — | — |
| HotpotQA | Graph 2-hop CE | — | — | — | — | — | — | — |
| HotpotQA | **Ours v2** | — | — | — | — | — | — | — |
| 2Wiki | Dense CE | — | — | — | — | — | — | — |
| 2Wiki | Graph 1-hop CE | — | — | — | — | — | — | — |
| 2Wiki | Graph 2-hop CE | — | — | — | — | — | — | — |
| 2Wiki | **Ours v2** | — | — | — | — | — | — | — |

Runtime phải đo từng method bằng process riêng. Evidence, prompt và output tokens phải được báo cáo tách biệt.

## 11. Protocol chạy

### 11.1 Giai đoạn pilot

- 200 QID cố định mỗi benchmark.
- Dùng để kiểm tra code, runtime và failure mode.
- Không dùng để chọn hyperparameter riêng cho từng dataset.

### 11.2 Giai đoạn final

- HotpotQA: full 7.405 câu.
- 2WikiMultiHopQA: full 12.576 câu.
- Retrieval chạy full cho mọi method.
- LLM generation chạy cùng evaluation set cho mọi method; nếu giới hạn chi phí buộc dùng subset thì phải dùng cùng paired QID và ghi rõ `n` trong bảng.
- `temperature=0`, cùng model và prompt.
- Paired bootstrap 10.000 resamples, seed 0.
- Mỗi result row ghi config SHA-256 và code revision.

## 12. Thứ tự thực hiện

1. Freeze Ours v2 và kiểm tra unit tests.
2. Chạy RQ2 trên n=200 để xác nhận graph-score operating point.
3. Chạy RQ1 retrieval full trên cả hai benchmark.
4. Chạy RQ1 LLM generation theo paired protocol.
5. Tính main table và paired bootstrap CI.
6. Chạy RQ3 budget sweep.
7. Chạy RQ4 component ablation.
8. Sinh RQ5 type/error breakdown từ output.
9. Đo RQ6 runtime/cost bằng process độc lập.

## 13. Tiêu chí kết luận

Ours v2 được xem là đạt pain point khi đồng thời:

1. Evidence tokens trung bình không vượt 200.
2. Giảm ít nhất 2× evidence tokens so với comparator chính.
3. Answer F1 đạt non-inferiority margin `−0.02` theo paired CI.
4. SP F1 hoặc SP Precision cải thiện rõ ràng.
5. Xu hướng không đảo chiều giữa Hotpot và 2Wiki.

Nếu graph score chỉ tăng SP metrics nhưng làm Answer F1 giảm vượt margin, paper phải trình bày đây là evidence–answer trade-off, không được claim graph score cải thiện chất lượng tổng thể.

## 14. Pilot hiện có — chỉ để tham khảo

| Dataset | Variant | SP Recall | SP F1 | Answer EM | Answer F1 | Tokens |
|---|---|---:|---:|---:|---:|---:|
| HotpotQA | Ours không graph score | **0.8809** | **0.4830** | 0.5450 | 0.6999 | 195.5 |
| HotpotQA | Ours graph-gated, best retrieval | 0.8793 | 0.4818 | — | — | 195.5 |
| 2Wiki | Ours không graph score | 0.8225 | 0.4121 | **0.4950** | **0.5940** | 192.9 |
| 2Wiki | Ours graph-gated, `λ_max=0.20` | **0.8238** | **0.4144** | 0.4900 | 0.5865 | **192.7** |

Pilot hiện chưa chứng minh graph score cải thiện Answer F1. Việc chọn graph score làm Ours v2 phải được xem là một giả thuyết cần kiểm chứng bằng RQ1/RQ2 final, không phải kết luận đã được dữ liệu xác nhận.
