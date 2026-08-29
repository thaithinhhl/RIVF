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

Canonical constants are `K=5`, `H=2`, `M=2`, `L=2`, `beta=0.30`, and
`B=200`. Graph structure proposes candidates; it is not a direct relevance
term in the main rank.
