# Bảng số liệu cho paper — Evidence-Efficient Graph Expansion for Multi-Hop QA

> **Ghi chú làm việc (không đưa vào paper):** File này gom các bảng "sạch", định dạng sẵn để dán vào bài paper (tiếng Anh, đúng quy ước academic). Nguồn số liệu chi tiết + lịch sử thực nghiệm đầy đủ nằm ở `version1.md`.
>
> - **HotpotQA**: số liệu thật, đã verify, **n=1000** (đã ổn định qua 3 lần mở rộng mẫu 200→350→1000, không còn dao động mạnh).
> - **2WikiMultiHopQA**: pipeline cuối cùng (title-prefix + reranker, α=0.6/budget=300) **chưa chạy lại** — các bảng dưới đây chỉ là khung/template giữ đúng cấu trúc cột để đối chiếu song song với HotpotQA, ô số liệu để trống (`—`). Không dùng số trong phần này cho tới khi có dữ liệu thật.

---



## Research Questions


| RQ      | Name                | Question                                                                                      | Experiment                               |
| ------- | ------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **RQ1** | Effectiveness       | Can selective graph expansion preserve supporting evidence while removing irrelevant context? | Main Comparison                          |
| **RQ2** | Efficiency          | How much retrieval context can be reduced without degrading downstream QA performance?        | Same-budget comparison + Trade-off curve |
| **RQ3** | Selection mechanism | How do semantic relevance and graph structure contribute to evidence selection?               | Ablation                                 |


---



## HotpotQA (Primary Benchmark)

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

*Caption gợi ý: "Supporting Fact Recall vs. context cost, 100-600 token range. Ours (blue) sits strictly above all baselines across the entire tested budget range."*

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

*Nhận định cho paper: scoring mechanism vượt trội rõ rệt so với random pruning (+0.382 SP Recall). Ở n=1000, semantic-only và blend (α=0.6) gần như tương đương (SP Recall bằng nhau, SP F1 semantic-only nhỉnh 0.003) — α=0.6 được giữ để phản ánh đúng đóng góp kết hợp 2 tín hiệu như thiết kế đề xuất, không đánh đổi hiệu năng đáng kể.*

**Bổ sung — phát hiện đáng đưa vào Discussion:** thay Top-K trên điểm gộp tuyến tính bằng chiến lược **interleave** (xen kẽ chọn từ ranking semantic-only và graph-only) cho kết quả trái chiều thú vị trên n=200: SP F1/Precision giảm (0.403→0.376 / 0.266→0.242) nhưng **EM tăng 0.525→0.540, Answer F1 tăng 0.667→0.685**. Phân tích sâu hơn (stratify theo semantic-sufficient/semantic-blind/unrecoverable — xem `version1.md` mục 6b) cho thấy đây không phải nhiễu ngẫu nhiên: blend hiện tại chỉ đạt 0.628 SP Recall trên nhóm "semantic-blind" (6.5% mẫu) dù về lý thuyết phải đạt gần 1.0. Kết luận giữ α=0.6 làm phương pháp chính (đúng định vị evidence-efficient), nhưng nêu rõ đây là trade-off có thật giữa evidence-cleanliness và answer-quality, không phải một chiều thắng tuyệt đối.

### Table 4 (bổ trợ) — Breakdown by Question Type


| Method      | Bridge (SP Recall / Answer F1) | Comparison (SP Recall / Answer F1) |
| ----------- | ------------------------------ | ---------------------------------- |
| Dense RAG   | 0.835 / 0.648                  | 0.983 / 0.780                      |
| Fixed 1-hop | 0.960 / 0.719                  | 0.984 / 0.721                      |
| Fixed 2-hop | 0.987 / 0.738                  | 0.992 / 0.726                      |
| **Ours**    | 0.898 / 0.702                  | 0.989 / 0.764                      |


*Dùng nếu cần giải thích vì sao comparison-type đạt SP Recall cao (nhưng ở n=1000 không còn tuyệt đối 1.0 ở method nào): các câu hỏi này nêu tên cả hai thực thể trực tiếp trong câu hỏi, và gold evidence thường là câu mở đầu (định nghĩa) của chính passage thực thể đó — dễ được cả bi-encoder lẫn reranker xếp hạng cao, nhưng không phải tuyệt đối với mẫu đủ lớn.*

---



## 2WikiMultiHopQA (Secondary Benchmark — Generalization)

> ⚠️ **Chưa có số liệu thật với pipeline cuối cùng.** Khung bảng dưới đây giữ đúng cấu trúc cột như HotpotQA để dễ đối chiếu song song khi có số liệu. Target operating point nên là **α=0.6, budget=300, reranker bật** — đồng bộ với cấu hình cuối cùng của HotpotQA (bảng cũ ở `version1.md` mục 9 dùng budget=600, KHÔNG reranker — không dùng cho paper).

*Dev set, n=? (đề xuất cùng n với HotpotQA để so sánh công bằng), 4 loại câu hỏi: comparison / inference / compositional / bridge_comparison (không có* `level`*).*

### Table 1' — Main Comparison (RQ1, generalization check)


| Method                    | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Context Tokens ↓ | # Nodes ↓ |
| ------------------------- | ----------- | -------------- | ------- | ----------- | ----------- | ---------------- | --------- |
| Dense RAG (300 tok)       | —           | —              | —       | —           | —           | —                | —         |
| Fixed 1-hop (unbounded)   | —           | —              | —       | —           | —           | —                | —         |
| Fixed 2-hop (unbounded)   | —           | —              | —       | —           | —           | —                | —         |
| **Ours** (α=0.6, 300 tok) | —           | —              | —       | —           | —           | —                | —         |




### Table 2' — Same-Budget Comparison (RQ2)


| Budget (tokens) | Dense RAG | Fixed 1-hop | Fixed 2-hop | **Ours** |
| --------------- | --------- | ----------- | ----------- | -------- |
| 150             | —         | —           | —           | —        |
| 300             | —         | —           | —           | —        |
| 600             | —         | —           | —           | —        |




### Figure 1' — Trade-off Curve (RQ2)

*(chưa có — cần chạy* `2wiki_budget_sweep` *với reranker rồi tái dùng* `make_figures.py`*)*

### Table 3' — Ablation Study (RQ3)


| Variant                  | α   | SP Recall ↑ | SP F1 ↑ |
| ------------------------ | --- | ----------- | ------- |
| Random pruning           | —   | —           | —       |
| Graph-only               | 0.0 | —           | —       |
| Semantic+Graph (blend)   | 0.6 | —           | —       |
| Semantic-only (reranker) | 1.0 | —           | —       |


---



## Việc cần làm trước khi chốt bảng (không đưa vào paper)

1. ~~Đợi mẫu HotpotQA n=1000 chạy xong~~ — xong, Table 1-4 đã cập nhật số liệu n=1000 (19/08/2026).
2. Chạy lại 2WikiMultiHopQA với pipeline cuối cùng (title-prefix + reranker, α=0.6/budget=300) → điền Table 1'-3' và vẽ Figure 1'.
3. Quyết định: đưa cả Table 2 (budget sweep) lẫn Figure 1 vào paper, hay chỉ giữ Figure 1 và rút Table 2 vào 2-3 số trong văn bản (do trùng thông tin, xem ghi chú dưới Table 2).
4. Table 4 (breakdown theo loại câu hỏi) là tùy chọn — chỉ thêm nếu còn chỗ hoặc reviewer cần giải thích thêm về SP Recall cao ở comparison-type.

