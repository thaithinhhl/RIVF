# Experimental Design — GRAFT-v1

Đây là protocol canonical đi cùng `pipeline.md`. Kết quả trong `results/` và
`result_v0.1/` được tạo trước GRAFT-v1 chỉ là development history.

## 1. Research claim

GRAFT giữ nguyên anchor–endpoint chains dưới hard ceiling 200 evidence tokens,
nhờ đó giữ Answer F1 cạnh tranh với dense/graph CE baselines dùng tối đa 600
evidence tokens.

Paper phải gọi đúng phạm vi là:

> Evidence selection and context compression over benchmark-provided candidate
> passages for multi-hop QA.

Không gọi đây là retrieval toàn Wikipedia hoặc open-domain RAG.

## 2. Benchmarks và dữ liệu

### 2.1 Hai benchmark chính

| Benchmark | Raw QIDs | Canonical valid QIDs | Development đã thấy | Evaluation holdout |
|---|---:|---:|---:|---:|
| HotpotQA dev-distractor | 7,405 | 7,404 | 397 | 7,007 |
| 2WikiMultiHopQA dev | 12,576 | 12,506 | 398 | 12,108 |

HotpotQA là benchmark chính; 2Wiki là external generalization benchmark vì có
bốn question types và nhiều bridge-comparison chains hơn.

### 2.2 Data hygiene

- Giữ nguyên raw JSON.
- Gộp passage trùng hoàn toàn trước khi tạo sentence candidates.
- Nếu cùng title nhưng nội dung khác nhau, preflight fail.
- Loại QID có supporting-fact index không tồn tại khỏi mọi method.
- Kiểm tra dataset SHA-256 trước run.
- Dense RAG và graph methods phải nhìn cùng canonical sentence identities.

Chạy:

```bash
python scripts/validate_benchmarks.py
python scripts/build_graft_manifests.py
```

Manifests:

- `data/manifests/hotpot_graft_v1_development.json`
- `data/manifests/hotpot_graft_v1_evaluation.json`
- `data/manifests/2wiki_graft_v1_development.json`
- `data/manifests/2wiki_graft_v1_evaluation.json`

Không được tune trên evaluation manifests.

## 3. Frozen method

GRAFT-v1 dùng chung trên hai benchmark:

```text
K=5, H=2, M=2, L=2, beta=0.30, B=200
```

- Graph chỉ proposal candidates/links.
- `CE(q,v)` xác minh node.
- Top-2 `CE(q+anchor,v)` links cho mỗi anchor là validated links.
- Utility pair cộng node(anchor), node(endpoint) và conditional link.
- Pair được giữ/bỏ atomic.
- Direct route dùng semantic budget-fill.
- Không PPR, graph residual hoặc adaptive alpha trong main rank.

## 4. Fair baselines

| Method | Candidate/ranking pipeline | Graph | Budget chính |
|---|---|---|---:|
| Dense RAG | CE trên mọi sentence | Không | 600 |
| Graph 1-hop | BGE seeds → 1-hop → CE | Có | 600 |
| Graph 2-hop | BGE seeds → 2-hop → CE | Có | 600 |
| GRAFT | 2-hop → node/link CE → atomic units | Có | 200 |

Mọi method dùng cùng QIDs, passages, generation model, prompt, tokenizer và
evaluation code. RQ2 chạy thêm tất cả baseline ở 200 tokens.

## 5. Research questions

| RQ | Câu hỏi | Artifact/config |
|---|---|---|
| RQ1 | GRAFT-200 có cạnh tranh với baseline-600? | Main retrieval + answer configs |
| RQ2 | Khi cùng 200 tokens, GRAFT có tốt hơn? | Cùng main configs, lấy rows B=200 |
| RQ3 | Quality–budget trade-off thế nào? | Budget configs |
| RQ4 | Thành phần nào tạo đóng góp? | Ablation configs |

## 6. RQ1 — Main comparison

### Retrieval/evidence evaluation

Chạy toàn bộ untouched evaluation holdout:

```bash
python -m rivf.experiments.run_experiment configs/final_hotpot_full_retrieval.yaml
python -m rivf.experiments.run_experiment configs/final_2wiki_full_retrieval.yaml
```

### Answer evaluation

Chạy 2.000 QID stratified từ evaluation holdout:

```bash
python -m rivf.experiments.run_experiment configs/graft_v1_hotpot_answer.yaml
python -m rivf.experiments.run_experiment configs/graft_v1_2wiki_answer.yaml
```

Primary endpoint là Answer F1. Báo thêm Answer EM, SP Recall, SP Precision, SP
F1, mean evidence tokens và mean selected nodes.

So sánh chính:

```text
GRAFT B=200 versus each baseline B=600
```

## 7. RQ2 — Same-budget control

Dùng đúng QIDs và output của answer configs, nhưng so sánh rows:

```text
GRAFT B=200 versus each baseline B=200
```

Không gọi thêm LLM nếu RQ1/RQ2 được chạy trong cùng config.

## 8. RQ3 — Budget sweep

Budgets:

```text
100, 150, 200, 300, 400, 600
```

Chạy 1.000 paired evaluation QIDs mỗi benchmark:

```bash
python -m rivf.experiments.run_experiment configs/graft_v1_hotpot_budget.yaml
python -m rivf.experiments.run_experiment configs/graft_v1_2wiki_budget.yaml
```

Vẽ:

1. Answer F1 theo mean evidence tokens.
2. SP F1 theo mean evidence tokens.
3. Answer F1 theo prompt tokens.

Không chỉ vẽ theo nominal budget; trục x chính phải là token thực dùng.

## 9. RQ4 — Component and edge ablation

Chạy 1.000 paired evaluation QIDs ở B=200:

```bash
python -m rivf.experiments.run_experiment configs/graft_v1_hotpot_ablation.yaml
python -m rivf.experiments.run_experiment configs/graft_v1_2wiki_ablation.yaml
```

| Variant | Thay đổi duy nhất/chính |
|---|---|
| Full GRAFT | Reference |
| `-GraphExpansion` | `H=0`, chỉ semantic seeds |
| `-Router` | Mọi question dùng pair policy |
| `-LinkVerification` | Bỏ conditional scoring/validated pairs |
| `-PairPreservation` | Giữ pair utility nhưng chỉ select endpoint |
| AdjacencyOnly | Chỉ local edges |
| Adjacency+Entity | Local + entity overlap |
| Adjacency+Title | Local + title mention |

Full GRAFT là dòng “all three edge types”. Edge ablation giải thích vì sao ba
quan hệ graph được chọn thay vì coi graph construction là heuristic tùy ý.

`-LinkVerification` làm mất validated pairs theo định nghĩa; vì vậy diễn giải
nó là ablation của toàn verification block, không diễn giải như một thay đổi CE
score hoàn toàn độc lập.

## 10. Statistical protocol

- Paired QIDs cho mọi comparison.
- Answer sample stratified theo question type.
- Paired bootstrap 95% CI, 10.000 resamples.
- Báo mean delta và CI.
- Báo per-type results kèm N.
- Không chọn checkpoint/hyperparameter dựa trên evaluation holdout.

Success criterion chính:

```text
lower_bound_95CI(AnswerF1_GRAFT200 - AnswerF1_baseline600) >= -0.02
```

Token compression phải đạt ít nhất 2× theo mean evidence tokens.

## 11. Reproducibility

Mỗi result row phải chứa:

- QID manifest.
- Dataset SHA-256.
- Config SHA-256.
- Git revision.
- Embedding/reranker model names.
- Requested and resolved LLM model.
- API response ID/system fingerprint/usage nếu có.
- Route, selected keys, validated-link metadata khi candidates được log.

Canonical config index nằm tại `configs/README.md`.

## 12. Execution order

1. `pytest` và benchmark preflight.
2. Full retrieval-only RQ1/RQ2.
3. Kiểm tra completeness, budget invariant và paired QIDs.
4. Answer n=2.000 RQ1/RQ2.
5. Budget sweep n=1.000.
6. Ablation n=1.000.
7. Freeze summary tables; không trộn `result_v0.1` với GRAFT-v1.

Để preflight retrieval mà không gọi API dù config bật LLM:

```bash
RIVF_SKIP_LLM=1 python -m rivf.experiments.run_experiment CONFIG.yaml
```
