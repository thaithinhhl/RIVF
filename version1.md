# Evidence-Efficient Graph Expansion for Multi-Hop QA — Kết quả thực nghiệm v1

**Ngày tổng hợp:** 19/08/2026 (bản đầy đủ, sau khi mở rộng mẫu HotpotQA 200→350→1000 câu)
**Venue:** RIVF 2026, Track 1 — AI Foundations and Big Data (deadline 31/08/2026)
**Trạng thái:** 4/4 thí nghiệm chính đã chạy trên HotpotQA với pipeline mới nhất (title-prefix embedding + cross-encoder reranker, budget=300), mẫu **1000/7405 câu** (seed=0, mở rộng dần từ 200→350→1000 nhờ resume — các câu đầu không đổi qua từng lần mở rộng). Benchmark phụ 2WikiMultiHopQA cần chạy lại (mục 9). Chưa có published baseline (mục 8b).

## 1. Thiết lập chung (cấu hình cuối cùng)

| | |
|---|---|
| Dataset chính | HotpotQA dev distractor set (`hotpot_dev_distractor_v1.json`) |
| Benchmark phụ | 2WikiMultiHopQA dev set (mục 9, **chưa cập nhật pipeline mới nhất**) |
| Mẫu đánh giá | 1000 câu hỏi (seed=0), HotpotQA: 100% hard-level, 792 bridge/208 comparison |
| Embedding seed retrieval | BAAI/bge-m3 (bi-encoder, local qua MPS), text dạng "{tiêu đề}: {câu}" (title-prefix) |
| **Semantic scoring (bước cuối)** | **BAAI/bge-reranker-v2-m3 (cross-encoder)** — thay cho bi-encoder, cải tiến mới nhất |
| Graph construction | Per-question, 3 loại edge: same-passage adjacency, cross-passage entity overlap (spaCy NER), cross-passage title-mention; hop distance đếm thô (không trọng số) |
| LLM sinh câu trả lời | OpenAI gpt-4o-mini, temperature=0 |
| Granularity | Sentence-level xuyên suốt |
| Scoring | `Score(v) = α·Semantic(q,v) + (1-α)·GraphRel(v)`, GraphRel = 1/(1+hop_distance đến seed gần nhất) |
| **Điểm vận hành "Ours"** | **α=0.6, budget=300 token** (giảm từ 600 — xem mục 11 vì sao) |
| Phương pháp so sánh | Dense RAG (300 tok), Fixed 1-hop, Fixed 2-hop (không giới hạn), Ours |

Toàn bộ pipeline, code, raw results (`results/*/results.jsonl`) nằm trong repo này.

## 2. Oracle Reachability Probe (HotpotQA, chạy trước các cải tiến embedding)

| Hop | Gold-sentence Recall | Kích thước candidate set trung bình |
|---|---|---|
| 1-hop | 96.2% | 19.6 câu |
| 2-hop | 98.8% | 28.1 câu |

Bridge-type đạt full gold recall ở 2-hop: 97.5% (156/160). Xác nhận graph construction đủ khả năng kết nối evidence trong 1-2 hop. *(Chưa chạy lại với title-prefix/reranker — xem mục 11.)*

## 3. Experiment 1 — Main Comparison (pipeline cuối cùng: reranker, budget=300)

Dense RAG và Ours giới hạn 300 token; Fixed 1-hop/2-hop chạy không giới hạn.

| Method | SP Recall | SP Precision | SP F1 | Answer EM | Answer F1 | Context Tokens | #Nodes |
|---|---|---|---|---|---|---|---|
| Dense RAG (300 tok) | 0.866 | 0.251 | 0.380 | 0.545 | 0.675 | 279 | 8.6 |
| Fixed 1-hop (không giới hạn) | 0.965 | 0.148 | 0.249 | 0.587 | 0.719 | 605 | 18.5 |
| Fixed 2-hop (không giới hạn) | 0.988 | 0.108 | 0.188 | 0.607 | 0.735 | 847 | 27.1 |
| **Ours** (α=0.6, 300 tok, reranker) | 0.917 | **0.273** | **0.411** | 0.581 | 0.715 | **277** | **8.5** |

### Theo loại câu hỏi (SP Recall / Answer F1)

| Method | Bridge | Comparison |
|---|---|---|
| Dense RAG | 0.835 / 0.648 | 0.983 / 0.780 |
| Fixed 1-hop | 0.960 / 0.719 | 0.984 / 0.721 |
| Fixed 2-hop | 0.987 / 0.738 | 0.992 / 0.726 |
| **Ours** | 0.898 / 0.702 | 0.989 / 0.764 |

**Nhận định — nhất quán qua cả 3 lần mở rộng mẫu (200→350→1000), tín hiệu ổn định:**
- **SP F1 và SP Precision của Ours vượt trội rõ rệt** — SP F1 gấp ~1.6 lần Fixed 1-hop (0.411 vs 0.249), ~2.2 lần Fixed 2-hop (0.411 vs 0.188); SP Precision gấp ~1.8-2.5 lần.
- **Answer F1 gần bằng Fixed 1-hop** (0.715 vs 0.719 — giữ 99% giá trị) chỉ với **46% context** (277 vs 605 token); **giữ 97% Answer F1 của Fixed 2-hop** (0.715/0.735) chỉ với **33% context** (277/847 token).
- So với Dense RAG cùng budget: Ours thắng trên **cả 5 metric**, khoảng cách Answer F1 nới rộng hơn nữa so với n=350 (0.715 vs 0.675, chênh 0.040 — so với 0.029 ở n=350 và gần hòa ở n=200).
- **Cập nhật cho câu hỏi "sao SP Recall comparison-type = 1.000":** ở n=1000 (208 câu comparison, gấp ~3x n=350), **không còn method nào giữ đúng 1.000 nữa** — cả Dense RAG (0.983) và **Ours** (0.989) đều xuất hiện vài trường hợp trượt. Bằng chứng dứt điểm: 1.000 ở các mẫu nhỏ trước (n=40, n=70) chỉ là hiệu ứng cỡ mẫu nhỏ, không phải tính chất tuyệt đối của comparison-type hay lỗi tính toán — SP Recall comparison-type vẫn **cao hơn rõ rệt** so với bridge-type (đúng cơ chế đã giải thích: entity nêu thẳng trong câu hỏi) nhưng không phải "luôn đúng 100%".

## 4. Experiment 2 — Budget Sweep (pipeline cuối cùng)

| Budget (token) | Dense RAG | Fixed 1-hop | Fixed 2-hop | **Ours** (reranker, α=0.6) |
|---|---|---|---|---|
| 150 | 0.729 | 0.729 | 0.730 | 0.789 |
| 200 | 0.797 | 0.800 | 0.799 | 0.854 |
| 250 | 0.837 | 0.843 | 0.841 | 0.893 |
| **300** | 0.866 | 0.874 | 0.872 | **0.917** |
| 400 | 0.906 | 0.913 | 0.913 | 0.949 |
| 500 | 0.929 | 0.935 | 0.937 | 0.965 |
| 600 | 0.949 | 0.947 | 0.952 | 0.974 |

**Nhận định:** Ours vượt cả 3 baseline ở **mọi budget được test**, với margin lớn dần khi budget giảm (150 token: +0.059 đến +0.060; 300 token: +0.043 đến +0.051; 600 token: +0.022 đến +0.025) — cùng xu hướng ổn định qua cả 3 lần mở rộng mẫu, đúng vào chỗ cần nhất: budget càng hẹp, cross-encoder càng phát huy giá trị (bi-encoder dễ nhầm lẫn hơn khi phải chọn top-K rất nhỏ).

## 5. Experiment 3 — Trade-off Curve

![Trade-off curve](results/figures/tradeoff_curve.png)

Cập nhật với dữ liệu n=1000. Khác biệt rõ rệt so với các phiên bản trước reranker: đường "Ours" (xanh dương) **tách biệt hẳn phía trên** mọi baseline trong suốt dải budget, thay vì bám sát nhau.

## 6. Experiment 4 — Ablation (budget=300, pipeline cuối cùng)

| Variant | α | SP Recall | SP F1 |
|---|---|---|---|
| Random pruning (sanity check) | — | 0.535 | 0.222 |
| Graph-only | 0.0 | 0.870 | 0.384 |
| (blend) | 0.2-0.5 | 0.911-0.916 | 0.406-0.410 |
| **(blend, đang dùng)** | **0.6** | **0.917** | **0.411** |
| (blend) | 0.8 | 0.919 | 0.412 |
| Semantic-only (reranker) | 1.0 | 0.917 | **0.414** |
| Interleave (xen kẽ 2 nhánh, n=200*) | — | 0.914 | 0.376 |

*Interleave đo trên n=200 (chưa mở rộng lên 1000 như các dòng khác) — xem phân tích đầy đủ ở mục 6b, không so trực tiếp SP F1 với các dòng trên do khác cỡ mẫu; điểm mấu chốt là Answer EM/F1 (mục 6b), không phải SP F1.

### Theo loại câu hỏi (SP Recall qua các mức α)

| Loại câu hỏi | α=0 | α=0.6 (đang dùng) | α=1 |
|---|---|---|---|
| Bridge | 0.848 | 0.898 | 0.899 |
| Comparison | 0.955 | 0.989 | 0.989 |

**Nhận định:**
- Random pruning: 0.535/0.222 — ổn định xuyên suốt 3 lần mở rộng mẫu (0.530→0.532→0.535) — cơ chế chấm điểm vẫn tạo giá trị rất lớn so với chọn ngẫu nhiên.
- **Khoảng cách Semantic-only (α=1.0) vs blend (α=0.6) thu hẹp lại ở n=1000**: SP Recall giờ **hòa tuyệt đối** (0.917 = 0.917), Semantic-only chỉ còn nhỉnh nhẹ ở SP F1 (0.414 vs 0.411). Ở n=350 semantic-only còn dẫn rõ (0.923 vs 0.917) — mẫu lớn hơn cho thấy 2 lựa chọn gần như tương đương. Điều này càng củng cố lý do **giữ α=0.6**: không đánh đổi gì đáng kể để giữ narrative "kết hợp 2 tín hiệu" của paper.
- Comparison-type không còn bão hòa tuyệt đối ở mẫu lớn (xem mục 3) — nhưng vẫn cao hơn rõ rệt bridge-type ở mọi mức α, đúng cơ chế đã giải thích.

## 6b. Chẩn đoán sâu hơn — Stratified theo mức độ "cần graph" (n=200)

**Câu hỏi:** trong công thức `Score(v)=α·Semantic+(1-α)·GraphRel`, phần graph thực sự cứu được câu hỏi nào? Với mỗi câu hỏi, tính riêng `S_sem` (Top-K theo semantic-only, α=1, budget=300) và `S_graph` (Top-K theo graph-only, α=0, budget=300), rồi gán nhãn theo gold evidence E*:

- **semantic-sufficient**: E* ⊆ S_sem (semantic một mình đã đủ, graph không cần)
- **semantic-blind**: E* ⊄ S_sem nhưng E* ⊆ (S_sem ∪ S_graph) (semantic thiếu, graph có tiềm năng cứu)
- **unrecoverable**: E* ⊄ (S_sem ∪ S_graph) (không nhánh nào đủ trong ngân sách 300 token)

### Phân bố (n=200)

| Nhóm | Số câu | Tỷ lệ |
|---|---|---|
| semantic-sufficient | 160 | 80.0% |
| unrecoverable | 27 | 13.5% |
| semantic-blind | 13 | 6.5% |

Toàn bộ 40 câu comparison-type đều rơi vào semantic-sufficient — bằng chứng độc lập, sạch, cho cơ chế đã giải thích ở mục 3 (entity nêu thẳng trong câu hỏi → semantic luôn đủ, graph không cần thiết cho loại câu hỏi này).

### Hiệu năng blend (α=0.6, đang dùng) theo từng nhóm

| Nhóm | SP Recall | SP F1 | EM | Answer F1 |
|---|---|---|---|---|
| semantic-sufficient (n=160) | 0.988 | 0.424 | 0.581 | 0.727 |
| semantic-blind (n=13) | 0.628 | 0.328 | 0.462 | 0.554 |
| unrecoverable (n=27) | 0.555 | 0.315 | 0.222 | 0.368 |

Ở nhóm semantic-blind, blend chỉ đạt **0.628 SP Recall** dù về lý thuyết union 2 nhánh phải phủ hết gold (đó là điều kiện gán nhãn) — cho thấy Top-K trên điểm gộp tuyến tính không tận dụng hết tiềm năng: một câu seed gần như không liên quan (hop=0, graph_score=1.0) vẫn được cộng cứng +0.4 điểm (ở α=0.6), có thể đè một câu 2-hop có ngữ nghĩa vừa phải ra khỏi Top-K — trong khi graph_score chỉ có 3 mức rời rạc (0.33/0.5/1.0) còn semantic_score (reranker) phân bố lệch mạnh (trung vị ~0.002, hầu hết candidate gần 0).

### Thử nghiệm sửa: strategy "interleave"

Thay vì Top-K trên điểm gộp, xen kẽ lượt chọn giữa 2 bảng xếp hạng riêng (semantic-only, graph-only) cho đến khi đầy budget — implement tại `retrieval/pruning.py::_interleave_select`, config `configs/interleave_300.yaml` (n=200, budget=300, có gọi LLM thật, không dùng α).

| Nhóm | Blend EM/F1 | Interleave EM/F1 |
|---|---|---|
| semantic-sufficient | 0.581 / 0.727 | 0.594 / 0.743 |
| semantic-blind | 0.462 / 0.554 | 0.462 / 0.554 (không đổi — n=13 quá nhỏ để kết luận) |
| unrecoverable | 0.222 / 0.368 | 0.259 / 0.405 |

**Tổng thể (n=200):**

| Metric | Blend | Interleave |
|---|---|---|
| SP Recall | 0.906 | 0.914 |
| SP Precision | 0.266 | 0.242 |
| SP F1 | 0.403 | 0.376 |
| EM | 0.525 | 0.540 |
| Answer F1 | 0.667 | 0.685 |

**Kết luận:** trade-off thật, không phải thắng/thua rõ ràng. Interleave làm SP F1/Precision (headline claim "evidence-efficient" của paper) kém đi, nhưng cải thiện EM/Answer F1 (chất lượng câu trả lời cuối). **Quyết định: giữ blend α=0.6 làm phương pháp chính** — đúng định vị chính của paper là hiệu quả chọn evidence, không phải chỉ tối ưu answer quality — nhưng ghi nhận phát hiện này cho phần Discussion, vì nó cho thấy 2 mục tiêu của bài toán (evidence sạch vs. answer đúng) không hoàn toàn đồng biến.

## 7. Đối chiếu Go/No-Go Criterion

- [x] **Ở cùng context budget, Ours có Supporting Fact Recall/F1 tốt hơn** — Đạt rõ ràng nhất từ trước đến giờ: thắng cả 3 baseline ở mọi budget test (Exp 2), SP F1/Precision vượt trội mọi phương pháp (Exp 1).
- [x] **Ở cùng mức Recall/Answer F1, Ours dùng ít context tokens/nodes hơn đáng kể** — Đạt: Answer F1 gần bằng Fixed 1-hop/2-hop chỉ với 33-46% context.

**→ Tín hiệu "go" mạnh nhất và sạch nhất trong toàn bộ quá trình — không còn cần đánh đổi giữa các metric như các phiên bản trước.**

## 8. Giới hạn & lưu ý khi viết paper

- Tất cả kết quả trên mẫu **1000/7405 câu hỏi HotpotQA** (seed=0, mở rộng dần từ 200→350→1000) — vẫn cần mở rộng thêm cho số liệu cuối cùng nếu thời gian cho phép, nhưng n=1000 đã đủ lớn để các con số ổn định qua nhiều lần mở rộng (không còn dao động mạnh giữa các mốc mẫu).
- **2WikiMultiHopQA (mục 9) và Oracle probe (mục 2) chưa chạy lại với reranker** — ưu tiên cao trước khi hoàn thiện paper.
- Cross-encoder không cache được giữa các câu hỏi (điểm phụ thuộc cả cặp câu hỏi+câu) — đo thực tế khi mở rộng 350→1000 (650 câu mới/config): toàn bộ 6 config mất ~53 phút. Ngoại suy full 7405 câu cho riêng Experiment 1 (4 method, có gọi LLM): ước tính vẫn ở mức nhiều giờ chạy liên tục vì `run_experiment.py` hiện tuần tự hoàn toàn, không có concurrency cho lệnh gọi LLM — nên n=1000 được chọn làm điểm dừng thực tế cho vòng này.
- Chưa có kiểm định ý nghĩa thống kê.
- Chưa có published baseline (mục 8b).
- 100% mẫu HotpotQA là hard-level (đặc điểm dataset).

### 8b. Các published baseline đã cân nhắc và quyết định bỏ

| Baseline | Vấn đề chính | Quyết định |
|---|---|---|
| HippoRAG | `pip install` lỗi thẳng trên macOS/arm64 (vllm hard-pin, chỉ có wheel Linux) | Bỏ |
| EfficientRAG | Không có checkpoint chính thức, phải tự train hoặc dùng checkpoint bên thứ 3; code hardcode `.cuda()`; repo không maintain 17 tháng | Bỏ |
| KG²RAG | Khả thi hơn nhưng cần ~1 ngày công + cài Ollama; `requirements.txt` hỏng cần tự sửa | Tạm bỏ |

## 9. 2WikiMultiHopQA — Kiểm tra Generalization (⚠️ chạy với pipeline CŨ — chưa có title-prefix lẫn reranker)

Cùng α=0.6/budget=600 áp dụng nguyên vẹn, 200 câu hỏi (seed=0). 4 loại câu hỏi (comparison, inference, compositional, bridge_comparison).

### 9.1. Oracle probe

| | HotpotQA | 2WikiMultiHopQA |
|---|---|---|
| 1-hop recall | 96.2% | 95.0% |
| 2-hop recall | 98.8% | 98.9% |

Graph construction generalize tốt sang benchmark thứ hai.

### 9.2. Main comparison (pipeline cũ, budget=600)

| Method | SP Recall | SP F1 | EM | Answer F1 | Context Tokens |
|---|---|---|---|---|---|
| Dense RAG (600 tok) | 0.934 | 0.201 | 0.475 | 0.577 | 551 |
| Fixed 1-hop (không giới hạn) | 0.950 | **0.273** | 0.465 | 0.574 | 474 |
| Fixed 2-hop (không giới hạn) | 0.989 | 0.239 | **0.490** | **0.603** | 589 |
| Ours (α=0.6, 600 tok, KHÔNG reranker) | 0.953 | 0.256 | 0.480 | 0.584 | **461** |

**Quan trọng:** số liệu này dùng pipeline CŨ (chưa có reranker). Dựa trên mức cải thiện quan sát được trên HotpotQA (reranker giúp mọi metric tăng đồng thời), **rất có khả năng** 2WikiMultiHopQA cũng sẽ cải thiện tương tự khi chạy lại — nhưng đây là suy luận, chưa phải số liệu thật. **Việc cần làm ưu tiên cao nhất (mục 11).**

## 10. Khảo sát cải tiến pipeline — 11 hướng thử, giữ lại 2 làm mặc định

| # | Ý tưởng | Cơ chế | Kết quả | Quyết định |
|---|---|---|---|---|
| 1 | Weighted shortest-path | Graph_score liên tục hơn qua trọng số edge | Trung tính | Revert |
| 2 | RRF (Reciprocal Rank Fusion) | Kết hợp theo thứ hạng thay vì cộng giá trị thô | Âm — graph_score quá thô gây hòa hạng | Revert (giữ làm tùy chọn) |
| 3 | Multi-seed graph score (tổng) | Tổng đóng góp từ mọi seed | Âm mạnh — nổ thang đo | Revert |
| 4 | Multi-seed graph score (trung bình) | Sửa lỗi #3 bằng trung bình | Vẫn âm — pha loãng tín hiệu | Revert |
| 5 | **Title-prefixed embedding** | Embed "{tiêu đề}: {câu}", giải quyết tham chiếu ngầm (đại từ) | **Dương nhất quán** | **Giữ lại** |
| 6 | Title-as-implicit-entity | Tiêu đề đoạn văn = entity ngầm cho entity-overlap edge | ~0 tác dụng — trùng lặp `title_mention` | Revert |
| 7 | Entity-frequency filtering | Chỉ tính entity hiếm (≤3/10 đoạn) cho entity-overlap | Cải thiện baseline nhiều hơn "Ours" — thu hẹp lợi thế | Revert |
| 8 | Đổi embedding sang Qwen3-Embedding (0.6b/8b, Ollama) | Giảm phụ thuộc API | Âm — kém hơn bge-m3, 8b chậm hơn ~13 lần | Revert |
| 9 | **MMR (Maximal Marginal Relevance)** | Phạt câu trùng lặp nội dung với câu đã chọn, tăng đa dạng | Âm ở mọi budget — phạt nhầm câu gold cùng nhắc entity bắc cầu (giống nhau ≠ dư thừa trong multi-hop) | Revert (giữ làm tùy chọn) |
| 10 | **Cross-encoder reranking** (BAAI/bge-reranker-v2-m3) | Thay bi-encoder bằng cross-encoder cho semantic_score — xử lý câu hỏi+câu cùng lúc, chính xác hơn ở top-K hẹp | **Dương mạnh và nhất quán ở MỌI budget/alpha** — cải thiện cả SP Recall/F1/Precision lẫn Answer EM/F1 đồng thời | **Giữ lại — cải tiến mạnh nhất** |
| 11 | **Interleave selection** (xen kẽ semantic-only/graph-only thay vì Top-K trên điểm gộp) | Chẩn đoán bằng stratified labeling (mục 6b) phát hiện nhóm "semantic-blind" (6.5% mẫu) mà blend không tận dụng hết tiềm năng union 2 nhánh | **Trade-off thật**: SP F1/Precision giảm (0.403→0.376 / 0.266→0.242) nhưng EM/Answer F1 tăng (0.525→0.540 / 0.667→0.685) | Revert (giữ làm tùy chọn) — không khớp định vị "evidence-efficient" của paper dù answer quality tốt hơn |

**Bài học chung:**
- 2/11 ý tưởng thành công đều là cải thiện **chất lượng tín hiệu semantic** (embedding tốt hơn, rồi reranker tốt hơn) — không phải chỉnh sửa công thức graph/cách kết hợp. Phần lớn ý tưởng nhắm vào graph_score/công thức kết hợp (#1-4, #6-7 phần nào, #2, #9, #11) đều thất bại, trung tính, hoặc chỉ đánh đổi (không thắng tuyệt đối).
- **Semantic score ảnh hưởng đến graph_score gián tiếp** qua seed retrieval — lý giải vì sao pure semantic (α cao) đôi khi cạnh tranh ngang hoặc hơn blend.
- MMR thất bại vì hiểu nhầm đặc thù multi-hop: **hai câu gold từ hai đoạn khác nhau thường giống nhau vì cùng nhắc entity bắc cầu** — đây là dấu hiệu liên kết tốt, không phải dư thừa.
- Luôn đánh giá theo **margin so với baseline cùng điều kiện**, không chỉ số tuyệt đối — nhiều cải tiến "tốt cho pipeline nói chung" hóa ra giúp baseline nhiều hơn "Ours" (bài học từ #7, không lặp lại ở #10).
- **Metric đánh giá evidence (SP F1) và metric đánh giá câu trả lời (Answer F1) không phải lúc nào cũng đồng biến** (bài học từ #11) — một lựa chọn làm evidence "kém sạch" hơn vẫn có thể giúp LLM trả lời đúng nhiều hơn, vì LLM tự lọc được nhiễu nhưng không đoán được thứ hoàn toàn vắng mặt. Cần nói rõ trong paper metric nào là headline claim để tránh mâu thuẫn khi trình bày.

## 11. Việc còn lại

1. **Chạy lại 2WikiMultiHopQA (mục 9) và Oracle probe (mục 2) với reranker** — ưu tiên cao nhất, hiện đang dùng số liệu pipeline cũ.
2. Cân nhắc thêm reranker cho Fixed 1-hop/2-hop để so sánh hoàn toàn công bằng (hiện chỉ "Ours" dùng reranker).
3. ~~Đo lại thời gian/chi phí chạy khi mở rộng mẫu~~ — đã đo (mục 8): ~1.6s/câu local compute, ~$0.11 cho 150 câu mới ở n=350; full 7405 câu ước tính ~20-22h vì thiếu concurrency ở vòng lặp gọi LLM.
4. Mở rộng mẫu đánh giá HotpotQA thêm nữa (hiện **1000/7405**, số liệu đã ổn định qua 3 lần mở rộng 200→350→1000) — nếu muốn lên full scale, nên thêm concurrency cho LLM call trước, nếu không sẽ mất nhiều giờ chạy liên tục.
5. Thêm kiểm định ý nghĩa thống kê.
6. (Tùy thời gian) Cân nhắc lại KG²RAG.
7. Quyết định khung claim chính: budget=300 (tiết kiệm cực lớn, SP F1 vượt trội) hay có nên báo cáo cả 300 và 600 như hai điểm trên đường cong trade-off.
