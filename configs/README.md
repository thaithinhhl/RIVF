# Experiment configs

## Canonical GRAFT-v1 configs

Only the following configs define the frozen main method described in
`pipeline.md` and `experiment_v2.md`:

- `final_hotpot_full_retrieval.yaml`
- `final_2wiki_full_retrieval.yaml`
- `graft_v1_hotpot_answer.yaml`
- `graft_v1_2wiki_answer.yaml`
- `graft_v1_hotpot_budget.yaml`
- `graft_v1_2wiki_budget.yaml`
- `graft_v1_hotpot_ablation.yaml`
- `graft_v1_2wiki_ablation.yaml`

All other YAML files are historical search/pilot configurations. They remain
in the repository only to reproduce prior exploratory results and must not be
reported as GRAFT-v1 final experiments.

`graft_v1_smoke.yaml` is a one-QID development-only implementation check. It
does not consume evaluation QIDs and is not a paper experiment.

## OpenRouter RQ1 run

`graft_v1_hotpot_rq1_openrouter_n1000.yaml` produces the five-row HotpotQA
RQ1 table on 1,000 paired evaluation QIDs: three baselines at B=600 and GRAFT
at B=600/B=200. It pins `baai/bge-m3`, `qwen/qwen3-reranker-8b`, and
`openai/gpt-4o-mini-2024-07-18` through OpenRouter. This is a new backend/model
protocol and must not be pooled with historical local BGE-reranker results.

`graft_v1_2wiki_rq1_openrouter_n1000.yaml` applies the identical five-row
OpenRouter RQ1 protocol to 1,000 paired, stratified 2WikiMultiHopQA evaluation
QIDs. Its method order and budgets match the reporting table: Dense RAG-600,
Graph 1-hop-600, Graph 2-hop-600, GRAFT-600, and GRAFT-200.

Canonical constants are `K=5`, `H=2`, `M=2`, `L=2`, `beta=0.30`, and
`B=200`. Graph structure proposes candidates; it is not a direct relevance
term in the main rank.
