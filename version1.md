# Evidence-Efficient Graph Expansion for Multi-Hop QA — Kết quả thực nghiệm v1

**Cập nhật:** 21/08/2026
**Kế hoạch chạy:** xem `experiment.md`

## 1. Mục tiêu và trạng thái

Paper kiểm chứng rằng **Ours dùng tối đa 200 evidence tokens nhưng vẫn giữ chất lượng câu trả lời cạnh tranh với baseline được dùng tối đa 600 tokens** trên HotpotQA và 2WikiMultiHopQA.

Pipeline hiện tại: BAAI/bge-m3 seed retrieval → graph expansion tối đa 2-hop → BAAI/bge-reranker-v2-m3 → label-free question router → conditional evidence-pair scoring → pair-aware pruning với hard ceiling 200 → gpt-4o-mini sinh câu trả lời. QA mode mặc định dùng single-chain; dual-chain chỉ dùng khi ưu tiên evidence recall.

Kết quả final-pipeline hiện là **pilot ghép cặp n=200** trên mỗi benchmark, chưa phải kết quả full để submit.

## 2. Research Questions

| RQ | Câu hỏi | Thí nghiệm | Trạng thái |
|---|---|---|---|
| **RQ1** | Ours-200 có giữ chất lượng khi baseline dùng 600 tokens? | Main Comparison | Pilot n=200 xong |
| **RQ2** | Lợi ích đến từ pipeline hay chỉ do budget khác nhau? | Same-budget Control | Cần chạy final budget 200 |
| **RQ3** | Hiệu quả thay đổi thế nào theo budget? | Budget Sweep/Pareto Curve | Cần chạy lại pipeline cuối |
| **RQ4** | Thành phần nào thực sự tạo cải thiện? | Ablation Study | Hoàn tất một phần |
| **RQ5** | Pipeline mạnh/yếu ở loại câu hỏi nào và vì sao? | Robustness/Error Analysis | Hoàn tất một phần |
| **RQ6** | Ours giảm bao nhiêu chi phí thực tế? | Runtime/Cost Analysis | Token đã có; runtime còn thiếu |

## 3. Thiết lập chung

| Thành phần | Thiết lập |
|---|---|
| Benchmark | HotpotQA dev distractor; 2WikiMultiHopQA dev |
| Granularity | Sentence-level |
| Seed retrieval | BAAI/bge-m3, text `title: sentence` |
| Semantic reranker | BAAI/bge-reranker-v2-m3 |
| Graph edge | Same-passage adjacency, entity overlap, title mention |
| Answer model | gpt-4o-mini, temperature=0 |
| Main budgets | Baseline 600; Ours 200 evidence tokens |
| Metrics | SP Recall/Precision/F1, Answer EM/F1, tokens, nodes |
| Statistics | Paired bootstrap 10.000 resamples, 95% CI |

Mọi method trong cùng comparison phải dùng cùng QID, seed, prompt và answer model. Gold supporting facts chỉ dùng để đánh giá/oracle probe, không đi vào pipeline.

---

## 4. RQ1 — Main Comparison: baseline 600 vs Ours 200

> Với 200 evidence tokens, Ours có giữ Answer F1 cạnh tranh với các baseline được dùng tối đa 600 tokens không?

### 4.1 HotpotQA, n=200

| Method | SP R | SP P | SP F1 | Answer EM | Answer F1 | Tokens | Nodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense BI — 600 | 0.9394 | 0.1300 | 0.2258 | 0.5300 | 0.6682 | 577.7 | 17.93 |
| Dense CE — 600 | 0.9643 | 0.1371 | 0.2372 | **0.5500** | 0.6996 | 577.8 | 17.47 |
| Graph 1-hop CE — 600 | 0.9548 | 0.1638 | 0.2744 | 0.5300 | 0.6730 | 496.4 | 15.03 |
| Graph 2-hop CE — 600 | **0.9687** | 0.1498 | 0.2555 | 0.5350 | 0.6916 | 542.1 | 16.48 |
| **Ours — 200** | 0.8809 | **0.3436** | **0.4830** | 0.5450 | **0.6999** | **195.5** | **6.34** |

- So với Dense CE: ít hơn **2.96× token**; ΔAnswer F1 `+0.0003`, CI95% `[-0.0417, 0.0407]`.
- So với Graph 2-hop: ít hơn **2.77× token**; ΔAnswer F1 `+0.0083`, CI95% `[-0.0341, 0.0504]`.
- Ours tăng SP Precision/F1 nhưng giảm SP Recall.

**Trả lời RQ1 trên Hotpot pilot:** Có. Answer F1 được giữ trong khi evidence context giảm gần 3 lần.

### 4.2 2WikiMultiHopQA, n=200

| Method | SP R | SP P | SP F1 | Answer EM | Answer F1 | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Dense BI — 600 | 0.9250 | 0.1146 | 0.2006 | 0.4700 | 0.5816 | 550.4 |
| Dense CE — 600 | 0.9450 | 0.1218 | 0.2123 | 0.4450 | 0.5602 | 552.2 |
| Graph 1-hop CE — 600 | 0.9563 | 0.1948 | 0.3157 | **0.5300** | **0.6372** | 388.4 |
| Graph 2-hop CE — 600 | **0.9588** | 0.1755 | 0.2884 | 0.4850 | 0.5939 | 433.2 |
| **Ours — 200** | 0.8225 | **0.2836** | **0.4121** | 0.4950 | 0.5940 | **192.9** |

- So với Graph 2-hop: ít hơn **2.25× token**; ΔAnswer F1 `+0.0001`, CI95% `[-0.0582, 0.0591]`.
- So với Graph 1-hop: ít hơn **2.01× token** nhưng Answer F1 thấp hơn `0.0432`, CI95% `[-0.0967, 0.0092]`.
- Ours có SP F1 cao nhất nhưng SP Recall thấp nhất.

**Trả lời RQ1 trên 2Wiki pilot:** Một phần. Ours giữ chất lượng so với Dense và Graph 2-hop, nhưng chưa giữ Answer F1 của Graph 1-hop.

### 4.3 Kết luận RQ1

Claim phù hợp hiện tại:

> Ours giảm evidence context khoảng 2–3 lần, giữ Answer F1 cạnh tranh và tăng độ chính xác của supporting evidence.

Chưa claim vượt SOTA hoặc giữ chất lượng trước mọi baseline. Cần chạy full HotpotQA 7.405 câu và 2Wiki 12.576 câu.

---

## 5. RQ2 — Same-budget Control

> Khi mọi method có cùng token budget, Ours có còn lợi thế không?

Controlled study hiện có dùng n=200, budget 300 và cùng reranker/LLM:

- **Hotpot:** Graph 2-hop CE semantic-only gần hòa Dense CE về retrieval nhưng Answer F1 tăng `+0.0200`, CI95% `[0.0025, 0.0425]`.
- **2Wiki:** Graph 2-hop CE tăng SP Recall `+0.0375`, SP F1 `+0.0334` và giảm 15.9 tokens so với Dense CE; Graph 1-hop có Answer F1 tốt nhất.
- **Alpha sweep:** α=1 tốt nhất trên cả hai benchmark; linear semantic/graph blend không tạo cực đại nội suy.

Study này hỗ trợ graph-bounded candidate expansion + CE pruning, nhưng chưa trực tiếp kiểm chứng Ours-200.

**Trạng thái RQ2:** Chưa hoàn tất. Cần chạy đủ năm methods với cùng hard ceiling 200 trên mỗi benchmark.

---

## 6. RQ3 — Budget Sweep và Pareto Frontier

> Ours có quality–token trade-off tốt hơn khi budget thay đổi không?

Budget cần chạy: `150, 200, 250, 300, 400, 500, 600`.

Bảng Hotpot n=1000 sau là **kết quả lịch sử** của pipeline α=0.6 và chưa bảo đảm reranker công bằng cho mọi baseline; không dùng làm bảng final.

| Budget | Dense RAG SP R | Graph 1-hop SP R | Graph 2-hop SP R | Ours SP R |
|---:|---:|---:|---:|---:|
| 150 | 0.729 | 0.729 | 0.730 | **0.789** |
| 200 | 0.797 | 0.800 | 0.799 | **0.854** |
| 250 | 0.837 | 0.843 | 0.841 | **0.893** |
| 300 | 0.866 | 0.874 | 0.872 | **0.917** |
| 400 | 0.906 | 0.913 | 0.913 | **0.949** |
| 500 | 0.929 | 0.935 | 0.937 | **0.965** |
| 600 | 0.949 | 0.947 | 0.952 | **0.974** |

![Trade-off curve lịch sử](results/figures/tradeoff_curve.png)

Final plot phải báo Answer F1–tokens và SP F1–tokens. Pareto curve dùng lại budget-sweep output, không tính là thí nghiệm riêng.

**Trạng thái RQ3:** Có tín hiệu lịch sử tích cực; cần chạy lại fair final-pipeline trên cả hai benchmark.

---

## 7. RQ4 — Ablation Study

> Thành phần nào tạo cải thiện và thành phần nào chỉ tạo trade-off?

### 7.1 Semantic–graph weighting

Controlled study cho thấy semantic-only α=1 tốt hơn blend α=0.6 trên cả hai benchmark. Contribution được hỗ trợ là **graph-bounded expansion + CE semantic pruning**, không phải inverse-hop linear blend.

| Variant lịch sử, Hotpot budget 300 | α | SP Recall | SP F1 |
|---|---:|---:|---:|
| Random pruning | — | 0.535 | 0.222 |
| Graph-only | 0.0 | 0.870 | 0.384 |
| Blend | 0.2–0.5 | 0.911–0.916 | 0.406–0.410 |
| Blend cũ | 0.6 | 0.917 | 0.411 |
| Semantic-only CE | 1.0 | 0.917 | **0.414** |

### 7.2 Single-chain vs dual-chain trên 2Wiki

- Single-chain: SP Recall/F1 `0.8225/0.4121`, Answer F1 `0.5940`, tokens `192.9`.
- Dual-chain: SP Recall/F1 `0.8500/0.4351`; bridge-comparison recall tăng `0.6341→0.7561`; tokens `192.6`.
- Gated chain-aware prompt chỉ đạt Answer F1 `0.5828`.

**Quyết định:** single-chain là QA mode; dual-chain là evidence-recall mode tùy chọn.

### 7.3 Adaptive PPR v2 trên Hotpot

| Method | SP R | SP P | SP F1 | Answer F1 | Tokens |
|---|---:|---:|---:|---:|---:|
| Graph 2-hop CE rank | 0.9158 | 0.2725 | 0.4113 | 0.6867 | 277.3 |
| Adaptive PPR, mass=0.50 | 0.8111 | 0.4552 | 0.5750 | 0.6363 | 146.8 |

PPR giảm context nhưng Answer F1 giảm `-0.0503`, CI95% `[-0.0961, -0.0065]`; không dùng làm headline method.

### 7.4 Graph-gated conditional rescue, n=200

Để đưa `graph_score` trở lại mà tránh linear blend, hệ thống thử semantic-weighted PPR và chỉ cộng graph residual cho node thuộc `S_graph − S_sem` đã được conditional CE xác nhận.

- Hotpot: không có weight `{0.05, 0.10, 0.15, 0.20}` nào vượt current retrieval; điểm tốt nhất đạt SP Recall/F1 `0.8793/0.4818` so với `0.8809/0.4830`.
- 2Wiki weight 0.20 tăng SP Recall/F1 `0.8225/0.4121 → 0.8238/0.4144` nhưng Answer F1 giảm `0.5940 → 0.5865`, Δ `-0.0075`, CI95% `[-0.0200, 0]`.

**Quyết định:** giữ làm ablation RQ4; không thay main single-chain pipeline vì evidence gain quá nhỏ và answer quality giảm.

### 7.5 Ablation final còn thiếu

Ở budget 200, loại lần lượt graph expansion, router, conditional reranking và pair-aware pruning; đồng thời so sánh single/dual-chain và semantic/graph-only.

**Trạng thái RQ4:** Hoàn tất một phần; đã loại linear blend/PPR khỏi claim chính nhưng chưa có component ablation đồng nhất cho pipeline cuối.

---

## 8. RQ5 — Robustness và Error Analysis

> Bottleneck nằm ở reachability, budget, pruning hay generation?

### 8.1 Oracle reachability lịch sử

| Benchmark | 1-hop gold recall | 2-hop gold recall |
|---|---:|---:|
| HotpotQA | 96.2% | 98.8% |
| 2Wiki | 95.0% | 98.9% |

Hotpot bridge đạt full-gold recall ở 2-hop cho 97.5% mẫu probe. Cần tái tính bằng pipeline cuối.

### 8.2 Budget feasibility trên 2Wiki

- 98.5% toàn bộ mẫu có gold-evidence cost ≤200 tokens.
- 92.7% `bridge_comparison` có gold-evidence cost ≤200 tokens.

Do đó SP Recall thấp chủ yếu là selection/routing bottleneck, không phải budget 200 bất khả thi.

### 8.3 Graph-need probe trên Hotpot, n=200

| Nhóm | Số câu | Tỷ lệ |
|---|---:|---:|
| Semantic-sufficient | 160 | 80.0% |
| Semantic-blind | 13 | 6.5% |
| Unrecoverable | 27 | 13.5% |

Toàn bộ 40 comparison questions thuộc semantic-sufficient. Interleave diagnostic tăng Answer F1 `0.667→0.685` nhưng giảm SP F1 `0.403→0.376`, xác nhận evidence quality và answer quality không luôn đồng biến.

Final analysis phải tách Hotpot bridge/comparison; bốn type của 2Wiki; ba nhóm graph-need; và failure do retrieval, routing, pruning hoặc generation.

**Trạng thái RQ5:** Bottleneck sơ bộ đã rõ; cần tái tính từ full/final output.

---

## 9. RQ6 — Runtime và Cost Analysis

> Ours tiết kiệm bao nhiêu context, latency và chi phí thực tế?

| Benchmark | Đối chứng | Baseline tokens | Ours tokens | Giảm |
|---|---|---:|---:|---:|
| Hotpot | Dense CE | 577.8 | 195.5 | 2.96× |
| Hotpot | Graph 2-hop | 542.1 | 195.5 | 2.77× |
| 2Wiki | Graph 1-hop | 388.4 | 192.9 | 2.01× |
| 2Wiki | Graph 2-hop | 433.2 | 192.9 | 2.25× |

Paper phải tách `evidence_tokens`, toàn bộ `prompt_tokens` và `output_tokens`. Runner mới đã ghi `prompt_tokens` cho các run mới.

Còn thiếu: retrieval/reranking latency, LLM latency, API cost, số nodes và peak memory. Phải đo từng method bằng process riêng để tránh cache liên-method.

**Trạng thái RQ6:** Compression ratio đã có; chưa kết luận được end-to-end latency/cost.

---

## 10. Kết luận theo RQ

| RQ | Kết luận hiện tại | Final? |
|---|---|---|
| RQ1 | Hotpot mạnh; 2Wiki cạnh tranh với Graph 2-hop nhưng thua Graph 1-hop | Pilot |
| RQ2 | Same-budget 300 có tín hiệu tốt cho graph-bounded CE pruning | Chưa |
| RQ3 | Budget curve lịch sử tích cực | Chưa |
| RQ4 | Single-chain hợp QA; dual-chain tăng recall; blend/PPR không hợp main | Một phần |
| RQ5 | Selection là bottleneck; SP và Answer có trade-off | Một phần |
| RQ6 | Evidence giảm khoảng 2–3× | Thiếu runtime/cost |

## 11. Giới hạn và việc còn lại

Giới hạn:

- Final-pipeline mới chạy n=200 trên mỗi benchmark.
- 2Wiki Graph 1-hop vẫn có Answer F1 cao hơn Ours.
- SP Recall của Ours thấp hơn baseline.
- Một số budget/oracle/ablation là kết quả pipeline cũ và đã được đánh dấu.
- Chưa có published GraphRAG baseline chạy thành công.

Thứ tự hoàn thiện:

1. **RQ1:** full main comparison trên cả hai benchmark.
2. **RQ2:** same-budget 200 với đủ năm methods.
3. **RQ3:** fair budget sweep trên cả hai benchmark.
4. **RQ4:** final component ablation ở budget 200.
5. **RQ5:** oracle, type breakdown và failure taxonomy từ final output.
6. **RQ6:** runtime, prompt/output tokens và API cost.
7. Paired bootstrap CI cho mọi chênh lệch headline.

## 12. Claim an toàn hiện tại

> Graph-bounded candidate expansion kết hợp cross-encoder conditional evidence selection có thể giảm evidence context khoảng 2–3 lần trong khi giữ chất lượng trả lời cạnh tranh trên multi-hop QA; hiệu quả phụ thuộc loại câu hỏi và tồn tại trade-off giữa evidence recall với answer quality.

Không claim vượt SOTA hoặc giữ nguyên chất lượng trên mọi benchmark cho đến khi RQ1–RQ6 hoàn tất theo protocol final.
