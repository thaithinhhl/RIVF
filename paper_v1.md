# Bảng số liệu cho paper — Evidence-Efficient Graph Expansion for Multi-Hop QA

> **Không phải GRAFT-v1 final result.** Tài liệu này ghi lại các pilot trước
> khi pipeline được freeze. Main method hiện tại nằm tại [`pipeline.md`](pipeline.md)
> và protocol final nằm tại [`experiment_v2.md`](experiment_v2.md).

> **Ghi chú làm việc (không đưa vào paper):** File này gom các bảng "sạch", định dạng sẵn để dán vào bài paper (tiếng Anh, đúng quy ước academic). Nguồn số liệu chi tiết + lịch sử thực nghiệm đầy đủ nằm ở `version1.md`.
>
> - **HotpotQA n=1000**: số liệu cũ đã verify, nhưng bảng chính dùng reranker riêng cho Ours nên **không còn được xem là controlled comparison**.
> - **Controlled study mới (20/08/2026)**: HotpotQA và 2WikiMultiHopQA đều đã chạy **n=200**, cùng seed, budget=300, reranker và LLM cho mọi CE variant. Đây là số liệu phải dùng để quyết định claim/method trước submission.
> - Kết luận mới: linear blend α=0.6 **không thắng semantic-only α=1.0** trên cả hai dataset. Không được giữ α=0.6 chỉ để phục vụ narrative.
> - **Adaptive PPR v2 (HotpotQA n=200)**: giảm context mạnh, nhưng điểm 146.8-token làm Answer F1 giảm có ý nghĩa thống kê. Chỉ trình bày như một efficiency operating point, không claim giữ nguyên chất lượng.
> - **Compression/transfer study mới (n=200 mỗi dataset)**: baseline được phép dùng 600 token, Ours có hard ceiling 200. Label-free question routing + conditional evidence-pair selection đã chạy đủ trên HotpotQA và 2Wiki. Đây là bảng efficiency chính; vẫn giữ bảng same-budget để bảo đảm fairness.

---



## Research Questions


| RQ      | Name                | Question                                                                                      | Experiment                               |
| ------- | ------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **RQ1** | Effectiveness       | Can selective graph expansion preserve supporting evidence while removing irrelevant context? | Main Comparison                          |
| **RQ2** | Efficiency          | How much retrieval context can be reduced without degrading downstream QA performance?        | Same-budget comparison + Trade-off curve |
| **RQ3** | Selection mechanism | How do semantic relevance and graph structure contribute to evidence selection?               | Ablation                                 |


---


## Submission-critical controlled study (n=200, seed=0)

*All methods use the same 300-token budget. “BI” uses BAAI/bge-m3; “CE” uses BAAI/bge-reranker-v2-m3. Fixed-hop CE ranks the hop-bounded graph candidate pool by the same cross-encoder as Ours. Answer generation uses gpt-4o-mini for every row.*

### HotpotQA — Fair same-budget comparison

| Method | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Tokens ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Dense BI | 0.859 | 0.246 | 0.375 | 0.525 | 0.655 | 279 |
| Dense CE | 0.913 | **0.273** | **0.411** | 0.520 | 0.667 | 279 |
| Graph expansion 1-hop + CE | 0.904 | 0.271 | 0.408 | 0.535 | 0.680 | **276** |
| **Graph expansion 2-hop + CE (semantic rank, α=1)** | **0.916** | **0.273** | **0.411** | **0.545** | **0.687** | 277 |
| Graph expansion 2-hop + BI (α=0.6) | 0.881 | 0.253 | 0.386 | 0.520 | 0.661 | 278 |
| Graph expansion 2-hop + CE (blend α=0.6) | 0.906 | 0.266 | 0.403 | 0.535 | 0.677 | 278 |

Paired bootstrap (10,000 resamples), 2-hop CE semantic rank vs. Dense CE: ΔSP Recall = +0.0028, 95% CI [-0.0058, 0.0113]; ΔSP F1 = +0.0001 [-0.0067, 0.0066]; ΔAnswer F1 = +0.0200 [0.0025, 0.0425]. Retrieval quality is statistically tied, while the 200-question pilot shows a small positive answer-quality difference.

### HotpotQA — 600-token baselines vs. 200-token Ours

*All rows use the same 200 questions, seed, candidate construction, and gpt-4o-mini answer generator. Baselines receive a 600-token ceiling. Ours uses a label-free question router: direct comparisons retain semantic CE selection, whereas chained questions use conditional second-hop reranking from the strongest anchor. Both branches obey a hard 200-token ceiling.*

| Method | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Tokens ↓ | Nodes ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense BI — 600 | 0.939 | 0.130 | 0.226 | 0.530 | 0.668 | 577.7 | 17.93 |
| Dense CE — 600 | 0.964 | 0.137 | 0.237 | **0.550** | **0.700** | 577.8 | 17.47 |
| Graph expansion 1-hop + CE — 600 | 0.955 | **0.164** | **0.274** | 0.530 | 0.673 | 496.4 | 15.03 |
| Graph expansion 2-hop + CE — 600 | **0.969** | 0.150 | 0.256 | 0.535 | 0.692 | 542.1 | 16.48 |
| **Ours — routed conditional pair, ceiling 200** | 0.881 | **0.344** | **0.483** | 0.545 | **0.700** | **195.5** | **6.34** |

Relative to Graph 2-hop CE, routed Ours uses 2.77× fewer selected-context tokens while mean Answer F1 is slightly higher (0.6999 vs. 0.6916). The paired Answer F1 difference is +0.0083 with 95% CI [-0.0341, 0.0504]; this pilot detects no loss but is not, by itself, a formal equivalence result. Ours has lower supporting-fact recall (-0.0878 [-0.1129, -0.0638]) but much higher supporting-fact F1 (+0.2275 [0.2097, 0.2446]). Relative to Dense CE, it uses 2.96× fewer tokens with essentially identical mean Answer F1 (+0.0003 [-0.0417, 0.0407]).

### 2WikiMultiHopQA — Fair transfer comparison

| Method | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Tokens ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Dense BI | 0.796 | 0.188 | 0.296 | 0.450 | 0.547 | 284 |
| Dense CE | 0.841 | 0.207 | 0.324 | 0.460 | 0.569 | 284 |
| **Graph expansion 1-hop + CE** | 0.878 | **0.239** | **0.368** | **0.495** | **0.599** | **259** |
| Graph expansion 2-hop + CE (semantic rank, α=1) | **0.879** | 0.231 | 0.358 | 0.485 | 0.584 | 268 |
| Graph expansion 2-hop + BI (α=0.6) | 0.844 | 0.221 | 0.341 | 0.460 | 0.555 | 267 |
| Graph expansion 2-hop + CE (blend α=0.6) | 0.865 | 0.227 | 0.352 | 0.485 | 0.582 | 266 |

Paired bootstrap, 2-hop CE semantic rank vs. Dense CE: ΔSP Recall = +0.0375 [0.0175, 0.0588]; ΔSP F1 = +0.0334 [0.0223, 0.0450]; Δtokens = -15.9 [-21.4, -10.8]. Answer F1 difference (+0.0150) is not significant at n=200. The graph-bounded candidate pool transfers positively for evidence retrieval, but the best 2Wiki operating point in this pilot is the simpler 1-hop CE variant.

### 2WikiMultiHopQA — 600-token baselines vs. 200-token routed Ours

| Method | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Tokens ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Dense BI — 600 | 0.925 | 0.115 | 0.201 | 0.470 | 0.582 | 550.4 |
| Dense CE — 600 | 0.945 | 0.122 | 0.212 | 0.445 | 0.560 | 552.2 |
| **Graph expansion 1-hop + CE — 600** | 0.956 | 0.195 | 0.316 | **0.530** | **0.637** | 388.4 |
| Graph expansion 2-hop + CE — 600 | **0.959** | 0.175 | 0.288 | 0.485 | 0.594 | 433.2 |
| **Ours — routed conditional pair, ceiling 200** | 0.823 | **0.284** | **0.412** | 0.495 | 0.594 | **192.9** |

Relative to Graph 2-hop CE, Ours uses 2.25× fewer tokens with effectively identical mean Answer F1 (difference +0.0001, 95% CI [-0.0582, 0.0591]). SP Recall is lower by -0.1363 [-0.1650, -0.1087], while SP F1 is higher by +0.1238 [0.1060, 0.1414]. Against Graph 1-hop CE, Ours uses 2.01× fewer tokens but mean Answer F1 is lower by -0.0432; its CI [-0.0967, 0.0092] includes zero at n=200. This transfer result supports context efficiency but also shows that the 200-token bottleneck is more difficult on 2Wiki than on HotpotQA.

**Two-chain ablation.** 2Wiki bridge-comparison questions average four gold facts but 92.7% of them still fit within 200 selected-evidence tokens, so the bottleneck is selection rather than an infeasible budget. Reserving one chain per question-grounded anchor raises overall SP Recall/F1 from 0.8225/0.4121 to 0.8500/0.4351 at 192.6 tokens; bridge-comparison recall rises from 0.6341 to 0.7561. However, its best gated-prompt run reaches only 0.5828 Answer F1, below the primary routed single-chain method's 0.5940. We therefore expose dual-chain packing as an evidence-recall operating mode, not the default QA method.

*Token accounting:* the table's `Tokens` column is selected evidence text, matching the retrieval-efficiency objective used throughout the experiments. New result rows additionally log total `prompt_tokens` (instructions, titles, question, and evidence) so end-to-end input cost can be audited separately.

### Controlled alpha ablation

| Dataset | α=0.0 SP R/F1 | α=0.2 | α=0.4 | α=0.6 | α=0.8 | α=1.0 SP R/F1 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | .875/.385 | .905/.401 | .907/.402 | .906/.403 | .910/.406 | **.916/.411** |
| 2Wiki | .864/.356 | .869/.354 | .868/.353 | .865/.352 | .865/.352 | **.879/.358** |

**Decision for the manuscript:** retire the α=0.6 linear blend as the primary method. The defensible result is graph-constrained candidate expansion followed by cross-encoder semantic pruning (α=1); it ties Dense CE retrieval and improves Answer F1 on HotpotQA, while improving evidence recall/F1 and reducing tokens on 2Wiki. The present evidence supports structural candidate filtering, not the proposed inverse-hop score.

### Adaptive PPR v2 — aggressive-efficiency ablation (HotpotQA)

*Same 200 questions and seed as the controlled HotpotQA study. The v2 pipeline uses CE semantic scores, query-level percentile normalization, PPR, a semantic-first adaptive α∈{1.0, 0.7, 0.4}, probability-mass stopping, a 300-token hard ceiling, and shortest-path closure. The operating point was selected using retrieval metrics before the LLM run.*

| Method | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Tokens ↓ | Nodes ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense BI | 0.859 | 0.245 | 0.375 | 0.525 | 0.655 | 279.0 | 8.57 |
| Dense CE | 0.913 | 0.273 | 0.411 | 0.520 | 0.667 | 278.7 | 8.34 |
| Graph expansion 1-hop + CE | 0.904 | 0.271 | 0.408 | 0.535 | 0.680 | 276.4 | 8.30 |
| Graph expansion 2-hop + CE (semantic rank) | **0.916** | 0.273 | 0.411 | **0.545** | **0.687** | 277.3 | 8.37 |
| **Adaptive PPR v2, mass=0.50** | 0.811 | **0.455** | **0.575** | 0.510 | 0.636 | **146.8** | **4.23** |

Adaptive PPR v2 removes 47.0% of selected context (1.89× fewer tokens) and substantially improves evidence precision/F1, but loses 0.0503 Answer F1. Paired bootstrap 95% CIs for v2 minus semantic rank are: ΔAnswer F1 = -0.0503 [-0.0961, -0.0065], ΔSP Recall = -0.1047 [-0.1309, -0.0791], ΔSP F1 = +0.1637 [0.1433, 0.1834], and Δtokens = -130.4 [-136.7, -124.1]. Therefore this point demonstrates a real controllable efficiency–quality trade-off, but it does **not** support the stronger claim of preserving answer quality. A less aggressive stopping point must be validated before being used as the primary result.


---



## HotpotQA (Original n=1000 result; reranker-confounded)

> ⚠️ Historical result only. Ours uses CE while the baselines below use BI. Do not present this table as the final fair main comparison; scale the controlled table above before submission.

*Dev-distractor set, n=1000 (seed=0), 100% hard-level, 792 bridge / 208 comparison. Pipeline: title-prefixed embedding (BAAI/bge-m3) + cross-encoder reranking (BAAI/bge-reranker-v2-m3) for "Ours".*

### Table 1 — Main Comparison (RQ1)


| Method                    | SP Recall ↑ | SP Precision ↑ | SP F1 ↑   | Answer EM ↑ | Answer F1 ↑ | Context Tokens ↓ | # Nodes ↓ |
| ------------------------- | ----------- | -------------- | --------- | ----------- | ----------- | ---------------- | --------- |
| Dense RAG (300 tok)       | 0.866       | 0.251          | 0.380     | 0.545       | 0.675       | 279              | 8.6       |
| Fixed 1-hop (unbounded)   | 0.965       | 0.148          | 0.249     | 0.587       | 0.719       | 605              | 18.5      |
| Fixed 2-hop (unbounded)   | 0.988       | 0.108          | 0.188     | 0.607       | 0.735       | 847              | 27.1      |
| **Ours** (α=0.6, 300 tok) | 0.917       | **0.273**      | **0.411** | 0.581       | 0.715       | **277**          | **8.5**   |


*Caption gợi ý: "Main comparison on HotpotQA dev-distractor (n=1000). Ours matches Fixed 1-hop's Answer F1 within 1% at 46% of its context, and Fixed 2-hop's within 3% at 33% of its context, while leading all methods on SP Precision/F1 and every Dense RAG metric at equal budget."*

### Table 2 — Same-Budget Comparison / Recall@Budget (RQ2)


| Budget (tokens) | Dense RAG | Fixed 1-hop | Fixed 2-hop | **Ours**  |
| --------------- | --------- | ----------- | ----------- | --------- |
| 150             | 0.729     | 0.729       | 0.730       | **0.789** |
| 200             | 0.797     | 0.800       | 0.799       | **0.854** |
| 250             | 0.837     | 0.843       | 0.841       | **0.893** |
| 300             | 0.866     | 0.874       | 0.872       | **0.917** |
| 400             | 0.906     | 0.913       | 0.913       | **0.949** |
| 500             | 0.929     | 0.935       | 0.937       | **0.965** |
| 600             | 0.949     | 0.947       | 0.952       | **0.974** |


*Metric: Supporting Fact Recall at equal context-token budget. Ours leads at every budget tested, with the margin widening as budget shrinks (+0.059–0.060 at 150 tok vs. +0.022–0.025 at 600 tok).*

> Gợi ý: nếu thiếu chỗ trong paper, có thể cắt bảng này còn 3 điểm (150/300/600) và dồn phần còn lại vào Figure 1 bên dưới — hai thứ đang thể hiện cùng một câu chuyện.



### Figure 1 — Trade-off Curve (RQ2)

File: `results/figures/tradeoff_curve.png` (trục X: context tokens 100-600 — cắt về đúng vùng đã sweep đều cho cả 4 method, trục Y: Supporting Fact Recall; 4 đường, không còn điểm sao unbounded — số liệu unbounded của Fixed-hop nằm ở Table 1).

*Historical caption only; do not use this figure as the final controlled comparison because its CE/BI setup differs across methods. Regenerate after the fair CE sweep is scaled.*

### Table 3 — Ablation Study (RQ3)

*Budget fixed at 300 tokens; varying the semantic/graph blend weight α in* `Score(v) = α·Semantic(q,v) + (1-α)·GraphRel(v)`*.*


| Variant                       | α          | SP Recall ↑ | SP F1 ↑   |
| ----------------------------- | ---------- | ----------- | --------- |
| Random pruning (sanity check) | —          | 0.535       | 0.222     |
| Graph-only                    | 0.0        | 0.870       | 0.384     |
| Semantic+Graph (blend, used)  | 0.6        | 0.917       | 0.411     |
| Semantic-only (reranker)      | 1.0        | 0.917       | **0.414** |
| Interleave (n=200)*           | —          | 0.914       | 0.376     |

\* *Interleave measured at n=200 (not yet scaled to n=1000 like the other rows) — see caption below; not directly comparable to the other rows' SP F1 due to sample size, and its point is the Answer-quality trade-off, not SP F1.*

*Nhận định đã supersede: random pruning vẫn xác nhận selection có giá trị, nhưng controlled study trên cả hai dataset chọn α=1.0. Không dùng lý do “giữ narrative kết hợp hai tín hiệu” để chọn α=0.6.*

**Bổ sung — phát hiện đáng đưa vào Discussion:** interleave cho thấy evidence cleanliness và answer quality không hoàn toàn đồng biến. Tuy nhiên controlled study mới cung cấp bằng chứng mạnh hơn rằng semantic-only α=1 là mặc định tốt hơn blend α=0.6; interleave chỉ nên giữ như diagnostic result.

### Table 4 (bổ trợ) — Breakdown by Question Type


| Method      | Bridge (SP Recall / Answer F1) | Comparison (SP Recall / Answer F1) |
| ----------- | ------------------------------ | ---------------------------------- |
| Dense RAG   | 0.835 / 0.648                  | 0.983 / 0.780                      |
| Fixed 1-hop | 0.960 / 0.719                  | 0.984 / 0.721                      |
| Fixed 2-hop | 0.987 / 0.738                  | 0.992 / 0.726                      |
| **Ours**    | 0.898 / 0.702                  | 0.989 / 0.764                      |


*Dùng nếu cần giải thích vì sao comparison-type đạt SP Recall cao (nhưng ở n=1000 không còn tuyệt đối 1.0 ở method nào): các câu hỏi này nêu tên cả hai thực thể trực tiếp trong câu hỏi, và gold evidence thường là câu mở đầu (định nghĩa) của chính passage thực thể đó — dễ được cả bi-encoder lẫn reranker xếp hạng cao, nhưng không phải tuyệt đối với mẫu đủ lớn.*

---



## 2WikiMultiHopQA (Secondary Benchmark — controlled pilot complete)

The final title-prefix + reranker pipeline has now been evaluated on n=200 under the same controlled design as HotpotQA. Use the controlled table near the top of this file. The older α=0.6/budget=600 result in `version1.md` remains historical only.


---



## Việc cần làm trước khi chốt bảng (không đưa vào paper)

1. ~~Đợi mẫu HotpotQA n=1000 chạy xong~~ — xong, Table 1-4 đã cập nhật số liệu n=1000 (19/08/2026).
2. ~~Chạy lại 2WikiMultiHopQA với pipeline cuối cùng~~ — controlled pilot n=200 đã xong; cần scale operating point đã sửa (α=1 hoặc 1-hop CE) nếu dùng làm bảng submission.
3. **Bắt buộc:** đổi method/claim khỏi linear blend α=0.6; controlled evidence không hỗ trợ lựa chọn này.
4. Scale fair CE comparison lên ít nhất n=1000 cho cả hai dataset; không scale lại các baseline BI đã bị supersede.
5. Quyết định: đưa cả Table 2 (budget sweep) lẫn Figure 1 vào paper, hay chỉ giữ Figure 1.
