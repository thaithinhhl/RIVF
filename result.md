# GRAFT-v1 — Experimental Results

> File này là template kết quả final cho pipeline trong [`pipeline.md`](pipeline.md)
> và protocol trong [`experiment_v2.md`](experiment_v2.md). Không chép số liệu
> từ `results/`, `result_v0.1/`, `paper_v1.md` hoặc `version1.md` nếu chúng không
> được tạo bởi canonical configs trong [`configs/README.md`](configs/README.md).

## 0. Run registry và provenance

Điền bảng này trước khi tổng hợp metric. Một run chỉ được đưa vào bảng final khi
đúng manifest, config hash và code revision đã freeze.

| Run | Dataset | Config | QID manifest | Target N | Completed N | Config SHA-256 | Git revision | Git clean? | Status |
|---|---|---|---|---:|---:|---|---|---|---|
| Main retrieval | HotpotQA | `final_hotpot_full_retrieval.yaml` | `hotpot_graft_v1_evaluation.json` | 7,007 | — | — | — | — | Pending |
| Main retrieval | 2Wiki | `final_2wiki_full_retrieval.yaml` | `2wiki_graft_v1_evaluation.json` | 12,108 | — | — | — | — | Pending |
| Main answer | HotpotQA | `graft_v1_hotpot_answer.yaml` | `hotpot_graft_v1_evaluation.json` | 2,000 | — | — | — | — | Pending |
| Main answer | 2Wiki | `graft_v1_2wiki_answer.yaml` | `2wiki_graft_v1_evaluation.json` | 2,000 | — | — | — | — | Pending |
| Budget sweep | HotpotQA | `graft_v1_hotpot_budget.yaml` | `hotpot_graft_v1_evaluation.json` | 1,000 | — | — | — | — | Pending |
| Budget sweep | 2Wiki | `graft_v1_2wiki_budget.yaml` | `2wiki_graft_v1_evaluation.json` | 1,000 | — | — | — | — | Pending |
| Ablation | HotpotQA | `graft_v1_hotpot_ablation.yaml` | `hotpot_graft_v1_evaluation.json` | 1,000 | — | — | — | — | Pending |
| Ablation | 2Wiki | `graft_v1_2wiki_ablation.yaml` | `2wiki_graft_v1_evaluation.json` | 1,000 | — | — | — | — | Pending |

### Frozen setup

| Thành phần | Giá trị |
|---|---|
| GRAFT constants | `K=5, H=2, M=2, L=2, β=0.30, B=200` |
| Embedding | `BAAI/bge-m3` |
| Cross-encoder | `BAAI/bge-reranker-v2-m3` |
| Answer model | `gpt-4o-mini-2024-07-18` |
| Temperature | `0` |
| Tokenizer | `cl100k_base` |
| Bootstrap | Paired, 10,000 resamples, 95% CI |

## RQ1 — GRAFT-200 so với baseline-600

**Research question:** GRAFT dùng tối đa 200 evidence tokens có giữ Answer F1
cạnh tranh với baseline dùng tối đa 600 tokens không?

### RQ1a. HotpotQA

| Method | Budget | N | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Mean evidence tokens ↓ | Mean prompt tokens ↓ | Mean nodes ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG | 600 | — | — | — | — | — | — | — | — | — |
| Graph 1-hop | 600 | — | — | — | — | — | — | — | — | — |
| Graph 2-hop | 600 | — | — | — | — | — | — | — | — | — |
| **GRAFT** | **200** | — | — | — | — | — | — | — | — | — |

### RQ1b. 2WikiMultiHopQA

| Method | Budget | N | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Mean evidence tokens ↓ | Mean prompt tokens ↓ | Mean nodes ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG | 600 | — | — | — | — | — | — | — | — | — |
| Graph 1-hop | 600 | — | — | — | — | — | — | — | — | — |
| Graph 2-hop | 600 | — | — | — | — | — | — | — | — | — |
| **GRAFT** | **200** | — | — | — | — | — | — | — | — | — |

**Kết luận RQ1:** _Điền sau khi hoàn thành hai bảng trên._

## RQ2 — Same-budget control ở 200 tokens

**Research question:** Khi mọi method đều bị giới hạn 200 tokens, lợi ích có đến
từ pipeline GRAFT thay vì chênh lệch budget không?

### RQ2a. HotpotQA

| Method | Budget | N | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Mean evidence tokens ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG | 200 | — | — | — | — | — | — | — |
| Graph 1-hop | 200 | — | — | — | — | — | — | — |
| Graph 2-hop | 200 | — | — | — | — | — | — | — |
| **GRAFT** | **200** | — | — | — | — | — | — | — |

### RQ2b. 2WikiMultiHopQA

| Method | Budget | N | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Mean evidence tokens ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG | 200 | — | — | — | — | — | — | — |
| Graph 1-hop | 200 | — | — | — | — | — | — | — |
| Graph 2-hop | 200 | — | — | — | — | — | — | — |
| **GRAFT** | **200** | — | — | — | — | — | — | — |

**Kết luận RQ2:** _Điền sau khi hoàn thành hai bảng trên._

## RQ3 — Budget sweep và Pareto frontier

Budgets: `100, 150, 200, 300, 400, 600`.

### RQ3a. HotpotQA — Answer F1

| Method | 100 | 150 | 200 | 300 | 400 | 600 |
|---|---:|---:|---:|---:|---:|---:|
| Dense RAG | — | — | — | — | — | — |
| Graph 1-hop | — | — | — | — | — | — |
| Graph 2-hop | — | — | — | — | — | — |
| **GRAFT** | — | — | — | — | — | — |

### RQ3b. 2Wiki — Answer F1

| Method | 100 | 150 | 200 | 300 | 400 | 600 |
|---|---:|---:|---:|---:|---:|---:|
| Dense RAG | — | — | — | — | — | — |
| Graph 1-hop | — | — | — | — | — | — |
| Graph 2-hop | — | — | — | — | — | — |
| **GRAFT** | — | — | — | — | — | — |

### RQ3c. HotpotQA — SP F1

| Method | 100 | 150 | 200 | 300 | 400 | 600 |
|---|---:|---:|---:|---:|---:|---:|
| Dense RAG | — | — | — | — | — | — |
| Graph 1-hop | — | — | — | — | — | — |
| Graph 2-hop | — | — | — | — | — | — |
| **GRAFT** | — | — | — | — | — | — |

### RQ3d. 2Wiki — SP F1

| Method | 100 | 150 | 200 | 300 | 400 | 600 |
|---|---:|---:|---:|---:|---:|---:|
| Dense RAG | — | — | — | — | — | — |
| Graph 1-hop | — | — | — | — | — | — |
| Graph 2-hop | — | — | — | — | — | — |
| **GRAFT** | — | — | — | — | — | — |

### RQ3e. Pareto points dùng token thực tế

| Dataset | Method | Nominal budget | Mean evidence tokens | Mean prompt tokens | Answer F1 | SP F1 | Pareto-optimal? |
|---|---|---:|---:|---:|---:|---:|---|
| HotpotQA | — | — | — | — | — | — | — |
| 2Wiki | — | — | — | — | — | — | — |

Figures:

- `figures/rq3_answer_f1_vs_evidence_tokens.png`
- `figures/rq3_answer_f1_vs_prompt_tokens.png`
- `figures/rq3_sp_f1_vs_evidence_tokens.png`

**Kết luận RQ3:** _Ghi operating point và Pareto frontier sau khi chạy._

## RQ4 — Component và graph-edge ablation

Tất cả variant dùng cùng QID và `B=200`.

### RQ4a. HotpotQA

| Variant | N | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Mean tokens ↓ | ΔAnswer F1 vs Full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full GRAFT** | — | — | — | — | — | — | — | 0.0000 |
| `-GraphExpansion` | — | — | — | — | — | — | — | — |
| `-Router` | — | — | — | — | — | — | — | — |
| `-LinkVerification` | — | — | — | — | — | — | — | — |
| `-PairPreservation` | — | — | — | — | — | — | — | — |
| AdjacencyOnly | — | — | — | — | — | — | — | — |
| Adjacency+Entity | — | — | — | — | — | — | — | — |
| Adjacency+Title | — | — | — | — | — | — | — | — |

### RQ4b. 2WikiMultiHopQA

| Variant | N | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Mean tokens ↓ | ΔAnswer F1 vs Full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full GRAFT** | — | — | — | — | — | — | — | 0.0000 |
| `-GraphExpansion` | — | — | — | — | — | — | — | — |
| `-Router` | — | — | — | — | — | — | — | — |
| `-LinkVerification` | — | — | — | — | — | — | — | — |
| `-PairPreservation` | — | — | — | — | — | — | — | — |
| AdjacencyOnly | — | — | — | — | — | — | — | — |
| Adjacency+Entity | — | — | — | — | — | — | — | — |
| Adjacency+Title | — | — | — | — | — | — | — | — |

### RQ4c. Paired ablation significance

| Dataset | Variant | Paired N | ΔSP F1 | 95% CI ΔSP F1 | ΔAnswer F1 | 95% CI ΔAnswer F1 | Interpretation |
|---|---|---:|---:|---|---:|---|---|
| HotpotQA | — | — | — | [—, —] | — | [—, —] | — |
| 2Wiki | — | — | — | [—, —] | — | [—, —] | — |

**Kết luận RQ4:** _Nêu component quan trọng và edge combination tốt nhất._

## Final claim checklist

- [ ] Mọi bảng dùng evaluation manifest, không dùng development QIDs.
- [ ] Mọi comparison có cùng paired QIDs.
- [ ] Không result row nào vượt nominal evidence budget.
- [ ] Answer N và retrieval N được báo riêng.
- [ ] `git_dirty=false` cho final runs.
- [ ] Dataset/config SHA và Git revision giống nhau trong từng run.
- [ ] Paired CI dùng 10.000 resamples.
- [ ] Không gọi result subset là full benchmark.
- [ ] Không gọi task này là open-domain/full-Wikipedia retrieval.

## Final conclusion

Điền sau khi hoàn tất RQ1–RQ4:

> GRAFT sử dụng trung bình **— evidence tokens**, giảm **—×** so với **—**, với
> chênh lệch Answer F1 **—** trên HotpotQA. Trên 2WikiMultiHopQA, mức giảm
> tương ứng là **—×** với chênh lệch Answer F1 **—**.
