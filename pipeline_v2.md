# GRAFT-v2 — Development Pipeline

GRAFT-v2 is a backward-compatible development successor to the frozen
GRAFT-v1 pipeline in [`pipeline.md`](pipeline.md). It must not replace v1 in
reported tables until its parameters are selected only on the development
manifests and the resulting configuration is frozen.

## Objective

Preserve the v1 principle

```text
graph proposes → semantics verifies → selection preserves
```

while fixing four observed failure modes:

1. direct comparisons do not benefit from graph-chain verification;
2. Top-L alone can call a harmful conditional link "validated";
3. static pair utility ignores marginal token cost and always fills the cap;
4. passage regrouping destroys the pair order produced by selection.

## Inference pipeline

```text
Question
  ↓
Label-free route + explicit passage-title grounding
  ├─ direct comparison
  │    all-sentence node CE
  │    one attribute witness per comparison entity
  │    stop when both entity roles are covered
  │
  └─ chained / bridge comparison
       graph-bounded proposal
       node CE + grounded anchors
       conditional-gain link verification
       one minimal chain (two for bridge comparison)
       semantic fallback only for missing witness roles
  ↓
Selection-order context serialization
  ↓
Answer model
```

The v2 router additionally recognizes plural and irregular relation nouns in
bridge comparisons (for example, “directors”); the frozen v1 router is left
unchanged for reproducibility.

## Conditional-gain verification

For a proposed anchor-endpoint link:

```text
gain(a,v) = CE(q + anchor, v) - CE(q, v)
```

A v2 link must be in the anchor's Top-L conditional ranking and satisfy:

```text
gain(a,v) >= conditional_link_min_gain
```

The development default is zero: adding the anchor may not reduce endpoint
relevance. GRAFT-v1 retains its original Top-L behavior.

## Minimal-witness compression

The hard budget is a ceiling, never a fill target. Route structure defines the
minimum witness:

| Route | Required witness |
|---|---|
| direct comparison | one fact from each of two grounded entities |
| chained reasoning | one validated anchor-endpoint chain |
| bridge comparison | two validated chains with distinct source anchors |

If link verification cannot construct the required witness, v2 falls back to
the smallest set of high-relevance facts from distinct passages. Optional
rescue is bounded by `witness_rescue_max_units`; the development default is
zero. Thus increasing `Bmax` alone cannot make v2 add evidence.

Within the required witness, units use marginal token-aware ranking:

For a unit `u` and current selected set `S`:

```text
ΔU(u | S) = new node utility + beta * newly completed link utility
             + coverage bonus

score(u | S) = ΔU(u | S) / incremental_tokens(u | S)^eta
```

Already-selected anchors receive neither duplicate node utility nor duplicate
token cost. `incremental_tokens` is computed from the compact serialized
context, including title headers. Sentence-text tokens remain the common hard
ceiling currency for fair comparison with existing baselines.

Development defaults:

| Parameter | Value |
|---|---:|
| `conditional_link_min_gain` | 0.0 |
| `token_cost_power` (`eta`) | 1.0 |
| `min_marginal_score` | 0.015 |
| `marginal_coverage_bonus` | 0.0 |
| `witness_min_titles` | 2 |
| `witness_chain_target` | 1 |
| `bridge_witness_chain_target` | 2 |
| `witness_rescue_max_units` | 0 |
| hard ceiling | 200 |

These are starting points, not final evaluation parameters.

## Prompt serialization

`compact_selection_order_prompt: true` preserves atomic witness order, emits a
passage title only when it changes, and does not add a reasoning hint. Selection
optimizes the token cost of this exact context representation.

## Quality-constrained objective

The development target is not merely to stay below 200 tokens. At the same
maximum budget as every baseline, v2 must minimize actual prompt tokens while
remaining non-inferior in Answer F1:

```text
minimize mean actual prompt tokens
subject to lower_95CI(AnswerF1_v2 - AnswerF1_reference) >= -0.02
```

Report the maximum budget as `Bmax` and actual evidence/prompt tokens
separately. Final claims require both lower actual usage and non-inferiority.

## Development protocol

Only these manifests may be used to select v2 parameters:

- `hotpot_graft_v1_development.json` (397 QIDs)
- `2wiki_graft_v1_development.json` (398 QIDs)

Paired five-way retrieval screens (Dense, Graph 1-hop, Graph 2-hop, GRAFT-v1,
and GRAFT-v2 all use `Bmax=200`):

```bash
python -m rivf.experiments.run_experiment configs/graft_v2_hotpot_development_openrouter.yaml
python -m rivf.experiments.run_experiment configs/graft_v2_2wiki_development_openrouter.yaml
```

Before a confirmatory run:

1. choose one v2 configuration using development QIDs only;
2. freeze config and code revision;
3. exclude already-inspected evaluation QIDs from the new test manifest;
4. report Answer F1, SP Recall, actual evidence/prompt tokens, and retrieval
   compute with paired confidence intervals.
