# Result v0.1 — target 200 questions per benchmark

> **Historical pilot, not GRAFT-v1.** Pipeline/config của các kết quả này khác
> main method đã freeze trong [`../pipeline.md`](../pipeline.md). Không dùng các
> bảng dưới đây làm final GRAFT-v1 claims.

All values are pilot/subsample results, not full-benchmark claims. `N < 200` means the OpenAI daily quota interrupted generation; retrieval-only rows deliberately leave Answer metrics blank.

## Run status

| Dataset | Complete paired QIDs | Target | Main status | RQ4 status |
|---|---:|---:|---|---|
| HotpotQA | 200 | 200 | Complete | Complete |
| 2Wiki | 181 | 200 | Partial — API quota | Retrieval complete; Answer pending |

## RQ1 — Main comparison: baseline 600 vs Ours 200

| Dataset | Method | Budget | N | SP Recall | SP Precision | SP F1 | Answer EM | Answer F1 | Evidence tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | Dense CE | 600 | 200 | 0.9850 | 0.1395 | 0.2418 | 0.6300 | 0.7470 | 578.2 |
| HotpotQA | Graph 1-hop CE | 600 | 200 | 0.9482 | 0.1603 | 0.2688 | 0.6000 | 0.7144 | 501.1 |
| HotpotQA | Graph 2-hop CE | 600 | 200 | 0.9625 | 0.1465 | 0.2499 | 0.6150 | 0.7281 | 551.2 |
| HotpotQA | Ours v2 | 200 | 200 | 0.8427 | 0.3336 | 0.4664 | 0.5850 | 0.7128 | 195.2 |
| 2Wiki | Dense CE | 600 | 181 | 0.9475 | 0.1250 | 0.2172 | 0.4309 | 0.5404 | 555.0 |
| 2Wiki | Graph 1-hop CE | 600 | 181 | 0.9420 | 0.1912 | 0.3108 | 0.4972 | 0.5961 | 401.4 |
| 2Wiki | Graph 2-hop CE | 600 | 181 | 0.9489 | 0.1723 | 0.2849 | 0.4972 | 0.6099 | 442.5 |
| 2Wiki | Ours v2 | 200 | 181 | 0.8097 | 0.2901 | 0.4156 | 0.4530 | 0.5616 | 193.4 |

### RQ1 paired non-inferiority

| Dataset | Comparator 600 | Δ Answer F1 | 95% CI | Non-inferior at −0.02 |
|---|---|---:|---|---|
| HotpotQA | Dense CE | -0.0342 | [-0.0833, 0.0133] | False |
| HotpotQA | Graph 1-hop CE | -0.0016 | [-0.0470, 0.0450] | False |
| HotpotQA | Graph 2-hop CE | -0.0153 | [-0.0596, 0.0292] | False |
| 2Wiki | Dense CE | 0.0212 | [-0.0437, 0.0863] | False |
| 2Wiki | Graph 1-hop CE | -0.0345 | [-0.0920, 0.0215] | False |
| 2Wiki | Graph 2-hop CE | -0.0483 | [-0.1065, 0.0096] | False |

## RQ2 — Same-budget control at 200 tokens

| Dataset | Method | N | SP Recall | SP Precision | SP F1 | Answer EM | Answer F1 | Evidence tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | Dense CE | 200 | 0.8542 | 0.3941 | 0.5221 | 0.5750 | 0.6953 | 179.7 |
| HotpotQA | Graph 1-hop CE | 200 | 0.8528 | 0.3926 | 0.5213 | 0.5600 | 0.6857 | 179.0 |
| HotpotQA | Graph 2-hop CE | 200 | 0.8579 | 0.3951 | 0.5236 | 0.5700 | 0.6970 | 180.5 |
| HotpotQA | Ours v2 | 200 | 0.8427 | 0.3336 | 0.4664 | 0.5850 | 0.7128 | 195.2 |
| 2Wiki | Dense CE | 181 | 0.7599 | 0.3018 | 0.4190 | 0.4088 | 0.5111 | 181.3 |
| 2Wiki | Graph 1-hop CE | 181 | 0.7710 | 0.3058 | 0.4252 | 0.4420 | 0.5407 | 180.6 |
| 2Wiki | Graph 2-hop CE | 181 | 0.7530 | 0.3016 | 0.4182 | 0.4475 | 0.5504 | 180.4 |
| 2Wiki | Ours v2 | 181 | 0.8097 | 0.2901 | 0.4156 | 0.4530 | 0.5616 | 193.4 |

## RQ3 — Answer F1 theo token budget

| Dataset | Method | 100 | 150 | 200 | 300 | 400 | 600 |
|---|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | Dense CE | 0.5449 | 0.6607 | 0.6953 | 0.7514 | 0.7429 | 0.7470 |
| HotpotQA | Graph 1-hop CE | 0.5411 | 0.6419 | 0.6857 | 0.7161 | 0.7046 | 0.7144 |
| HotpotQA | Graph 2-hop CE | 0.5509 | 0.6569 | 0.6970 | 0.7316 | 0.7298 | 0.7281 |
| HotpotQA | Ours v2 | 0.6276 | 0.6775 | 0.7128 | 0.7300 | 0.7247 | 0.7234 |
| 2Wiki | Dense CE | 0.4476 | 0.5072 | 0.5111 | 0.5499 | 0.5418 | 0.5404 |
| 2Wiki | Graph 1-hop CE | 0.4592 | 0.5158 | 0.5407 | 0.5610 | 0.5544 | 0.5961 |
| 2Wiki | Graph 2-hop CE | 0.4429 | 0.5051 | 0.5504 | 0.5603 | 0.5689 | 0.6099 |
| 2Wiki | Ours v2 | 0.5216 | 0.5354 | 0.5616 | 0.5806 | 0.5923 | 0.6306 |

## RQ3 — SP F1 theo token budget

| Dataset | Method | 100 | 150 | 200 | 300 | 400 | 600 |
|---|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | Dense CE | 0.6231 | 0.5976 | 0.5221 | 0.4137 | 0.3364 | 0.2418 |
| HotpotQA | Graph 1-hop CE | 0.6268 | 0.5883 | 0.5213 | 0.4106 | 0.3388 | 0.2688 |
| HotpotQA | Graph 2-hop CE | 0.6252 | 0.5918 | 0.5236 | 0.4142 | 0.3379 | 0.2499 |
| HotpotQA | Ours v2 | 0.5636 | 0.5164 | 0.4664 | 0.3878 | 0.3213 | 0.2433 |
| 2Wiki | Dense CE | 0.5362 | 0.4857 | 0.4190 | 0.3285 | 0.2804 | 0.2172 |
| 2Wiki | Graph 1-hop CE | 0.5387 | 0.4898 | 0.4252 | 0.3602 | 0.3370 | 0.3108 |
| 2Wiki | Graph 2-hop CE | 0.5369 | 0.4876 | 0.4182 | 0.3517 | 0.3177 | 0.2849 |
| 2Wiki | Ours v2 | 0.5126 | 0.4623 | 0.4156 | 0.3475 | 0.3155 | 0.2843 |

Figures: [`rq3_answer_f1_pareto.png`](rq3_answer_f1_pareto.png) and [`rq3_sp_f1_pareto.png`](rq3_sp_f1_pareto.png).

## RQ4 — Component ablation at 200 tokens

| Dataset | Variant | N | SP Recall | SP Precision | SP F1 | Answer EM | Answer F1 | Tokens | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| HotpotQA | Ours v2 | 200 | 0.8427 | 0.3336 | 0.4664 | 0.5850 | 0.7128 | 195.2 | Complete |
| HotpotQA | -GraphScore | 200 | 0.8498 | 0.3348 | 0.4688 | 0.5700 | 0.6993 | 195.3 | Complete |
| HotpotQA | Uniform PPR | 200 | 0.8398 | 0.3326 | 0.4650 | 0.5450 | 0.6750 | 195.1 | Complete |
| HotpotQA | -DisagreementGate | 200 | 0.8411 | 0.3338 | 0.4664 | 0.5650 | 0.6950 | 195.0 | Complete |
| HotpotQA | -Router | 200 | 0.8453 | 0.3327 | 0.4665 | 0.5600 | 0.6966 | 195.3 | Complete |
| HotpotQA | -ConditionalScoring | 200 | 0.8615 | 0.3415 | 0.4776 | 0.5800 | 0.6937 | 195.5 | Complete |
| HotpotQA | -PairPreservation | 200 | 0.8282 | 0.3279 | 0.4584 | 0.5600 | 0.6911 | 195.2 | Complete |
| HotpotQA | Graph-only | 200 | 0.7959 | 0.3647 | 0.4873 | 0.4950 | 0.6212 | 179.3 | Complete |
| 2Wiki | Ours v2 | 181 | 0.8097 | 0.2901 | 0.4156 | 0.4530 | 0.5616 | 193.4 | Partial — API quota |
| 2Wiki | -GraphScore | 200 | 0.7978 | 0.2903 | 0.4132 | — | — | 193.6 | Retrieval-only; Answer pending |
| 2Wiki | Uniform PPR | 200 | 0.7990 | 0.2899 | 0.4131 | — | — | 193.6 | Retrieval-only; Answer pending |
| 2Wiki | -DisagreementGate | 200 | 0.8003 | 0.2905 | 0.4141 | — | — | 193.7 | Retrieval-only; Answer pending |
| 2Wiki | -Router | 200 | 0.8015 | 0.2905 | 0.4139 | — | — | 193.5 | Retrieval-only; Answer pending |
| 2Wiki | -ConditionalScoring | 200 | 0.7653 | 0.2713 | 0.3909 | — | — | 193.7 | Retrieval-only; Answer pending |
| 2Wiki | -PairPreservation | 200 | 0.7940 | 0.2858 | 0.4088 | — | — | 193.6 | Retrieval-only; Answer pending |
| 2Wiki | Graph-only | 200 | 0.7290 | 0.3120 | 0.4205 | — | — | 180.6 | Retrieval-only; Answer pending |

## RQ5 — Robustness and error analysis

### Breakdown theo question type

| Dataset | Type | N | SP Recall | SP F1 | Answer EM | Answer F1 | Tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | bridge_comparison | 37 | 0.6365 | 0.4801 | 0.4865 | 0.5010 | 195.8 |
| 2Wiki | comparison | 45 | 0.9556 | 0.4543 | 0.8667 | 0.8667 | 190.0 |
| 2Wiki | compositional | 78 | 0.8462 | 0.3826 | 0.2821 | 0.4187 | 193.9 |
| 2Wiki | inference | 21 | 0.6667 | 0.3419 | 0.1429 | 0.5454 | 194.5 |
| HotpotQA | bridge | 160 | 0.8266 | 0.4559 | 0.5500 | 0.7010 | 195.5 |
| HotpotQA | comparison | 40 | 0.9075 | 0.5085 | 0.7250 | 0.7597 | 194.2 |

### Mức độ cần graph

| Dataset | Group | N | SP Recall | SP F1 | Answer F1 | Tokens |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | Semantic-blind | 21 | 0.7286 | 0.4155 | 0.4019 | 195.9 |
| 2Wiki | Semantic-sufficient | 91 | 1.0000 | 0.4626 | 0.6795 | 191.7 |
| 2Wiki | Unrecoverable | 69 | 0.5833 | 0.3537 | 0.4547 | 194.8 |
| HotpotQA | Semantic-blind | 15 | 0.7444 | 0.4583 | 0.7022 | 197.7 |
| HotpotQA | Semantic-sufficient | 140 | 0.9385 | 0.5063 | 0.7945 | 194.8 |
| HotpotQA | Unrecoverable | 45 | 0.5778 | 0.3450 | 0.4620 | 195.9 |

### Oracle reachability

| Dataset | N | 1-hop gold recall | 2-hop gold recall | Full gold reachable | Mean gold tokens | Gold fit ≤200 |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | 181 | 0.9641 | 0.9779 | 0.9558 | 84.8 | 0.9945 |
| HotpotQA | 200 | 0.9515 | 0.9708 | 0.9350 | 82.1 | 1.0000 |

### Graph rescue diagnostic

| Dataset | N | Activated | Correct rescue | False rescue | Mean gold added | Mean gold displaced |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | 181 | 0.1050 | 0.0276 | 0.0884 | 0.0276 | 0.0110 |
| HotpotQA | 200 | 0.1100 | 0.0050 | 0.1100 | 0.0050 | 0.0200 |

Failure counts are multi-label and stored in [`rq5_failures.csv`](rq5_failures.csv).

## RQ6 — Efficiency and cost

| Dataset | Method | Budget | N | Evidence tok | Prompt tok | Output tok | Nodes | Retrieval s | LLM s | USD/question |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | Dense CE | 600 | 200 | 578.2 | 717.6 | 3.6 | 17.6 | 1.111 | 0.753 | 0.000110 |
| HotpotQA | Graph 1-hop CE | 600 | 200 | 501.1 | 633.6 | 3.4 | 15.2 | 0.221 | 0.750 | 0.000097 |
| HotpotQA | Graph 2-hop CE | 600 | 200 | 551.2 | 685.6 | 3.5 | 16.8 | 0.030 | 0.740 | 0.000105 |
| HotpotQA | Ours v2 | 200 | 200 | 195.2 | 304.5 | 3.6 | 6.3 | 0.595 | 0.755 | 0.000048 |
| 2Wiki | Dense CE | 600 | 181 | 555.0 | 694.2 | 3.9 | 19.0 | 0.812 | 0.844 | 0.000106 |
| 2Wiki | Graph 1-hop CE | 600 | 181 | 401.4 | 519.4 | 3.9 | 13.0 | 0.111 | 1.073 | 0.000080 |
| 2Wiki | Graph 2-hop CE | 600 | 181 | 442.5 | 563.3 | 3.8 | 14.6 | 0.001 | 1.647 | 0.000087 |
| 2Wiki | Ours v2 | 200 | 181 | 193.4 | 297.0 | 4.0 | 6.7 | 0.345 | 1.196 | 0.000047 |

RQ6 latency hiện là instrumented pilot trong shared run. Trước khi dùng làm final paper claim, cần đo lại từng method bằng process riêng như protocol trong `experiment_v2.md`.

## Reproducibility artifacts

- Full aggregate: [`aggregate_metrics.csv`](aggregate_metrics.csv)
- Paired bootstrap: [`rq1_paired_bootstrap_ci.csv`](rq1_paired_bootstrap_ci.csv)
- RQ5 detail: [`rq5_type.csv`](rq5_type.csv), [`rq5_graph_need.csv`](rq5_graph_need.csv), [`rq5_oracle.csv`](rq5_oracle.csv), [`rq5_rescue.csv`](rq5_rescue.csv), [`rq5_failures.csv`](rq5_failures.csv)

## Accounting

Cost uses gpt-4o-mini standard rates: $0.15/1M input and $0.6/1M output tokens. Evidence, prompt, and output tokens are reported separately.
