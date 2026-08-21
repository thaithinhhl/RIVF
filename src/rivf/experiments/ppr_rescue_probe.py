"""Gold-aware diagnostic: can PPR rescue evidence missed by semantic ranking?

This is an analysis tool, not an inference component. Gold supporting facts
are used only to test the premise behind adaptive semantic/graph fusion.

Usage:
    python -m rivf.experiments.ppr_rescue_probe --n-questions 200 --top-k 9
"""

import argparse
import json
import random
from dataclasses import dataclass

from rivf.data.hotpotqa import load_hotpotqa
from rivf.data.schema import QAItem
from rivf.methods.base import Candidate
from rivf.retrieval.pruning import prune
from rivf.retrieval.scoring import generate_candidates


@dataclass
class RescueObservation:
    candidate_gold: int
    gold_total: int
    semantic_gold: int
    graph_gold: int
    union_gold: int
    recoverable_semantic_misses: int
    ppr_rescued_misses: int
    random_rescued_misses: float
    semantic_complete: bool
    union_complete: bool
    missing_gold_ppr_percentiles: list[float]
    disagreement: float
    semantic_top2_mean: float
    semantic_kth_score: float
    ppr_concentration: float
    seed_miss_fraction: float


def evaluate_candidates(
    item: QAItem,
    candidates: list[Candidate],
    *,
    budget_tokens: int | None,
    top_k: int | None,
    random_trials: int = 100,
    rng_seed: int = 0,
) -> RescueObservation:
    semantic = prune(
        candidates, alpha=1.0, strategy="topk", budget_tokens=budget_tokens, top_k=top_k
    )
    graph = prune(
        candidates, alpha=0.0, strategy="topk", budget_tokens=budget_tokens, top_k=top_k
    )
    semantic_keys = {s.key for s in semantic}
    graph_keys = {s.key for s in graph}
    candidate_keys = {c.sentence.key for c in candidates}
    gold = set(item.supporting_facts)
    recoverable_misses = (gold & candidate_keys) - semantic_keys

    semantic_order = sorted(candidates, key=lambda c: c.semantic_score or 0.0, reverse=True)

    graph_order = sorted(
        candidates,
        key=lambda c: (c.graph_score or 0.0, c.semantic_score or 0.0),
        reverse=True,
    )
    graph_rank = {c.sentence.key: rank for rank, c in enumerate(graph_order)}
    denominator = max(1, len(graph_order) - 1)
    # 1.0 means the missing gold fact is first in the PPR ranking; 0.0 last.
    percentiles = [1.0 - graph_rank[key] / denominator for key in recoverable_misses]
    union_keys = semantic_keys | graph_keys
    jaccard = len(semantic_keys & graph_keys) / len(union_keys) if union_keys else 1.0
    top2 = semantic_order[:2]
    semantic_top2_mean = sum(c.semantic_score or 0.0 for c in top2) / len(top2) if top2 else 0.0
    selection_size = max(1, len(semantic_keys))
    semantic_kth_score = semantic_order[min(selection_size, len(semantic_order)) - 1].semantic_score or 0.0
    graph_values = sorted((c.graph_score or 0.0 for c in candidates), reverse=True)
    median_graph = graph_values[len(graph_values) // 2] if graph_values else 0.0
    ppr_concentration = (graph_values[0] - median_graph) if graph_values else 0.0
    seeds = [c for c in candidates if c.hop_distance == 0]
    seed_miss_fraction = (
        sum(c.sentence.key not in semantic_keys for c in seeds) / len(seeds) if seeds else 0.0
    )

    rng = random.Random(rng_seed)
    random_rescued = []
    graph_selection_size = len(graph_keys)
    candidate_list = list(candidate_keys)
    for _ in range(random_trials):
        random_keys = set(rng.sample(candidate_list, min(graph_selection_size, len(candidate_list))))
        random_rescued.append(len(recoverable_misses & random_keys))

    return RescueObservation(
        candidate_gold=len(gold & candidate_keys),
        gold_total=len(gold),
        semantic_gold=len(gold & semantic_keys),
        graph_gold=len(gold & graph_keys),
        union_gold=len(gold & (semantic_keys | graph_keys)),
        recoverable_semantic_misses=len(recoverable_misses),
        ppr_rescued_misses=len(recoverable_misses & graph_keys),
        random_rescued_misses=sum(random_rescued) / len(random_rescued),
        semantic_complete=gold <= semantic_keys,
        union_complete=gold <= (semantic_keys | graph_keys),
        missing_gold_ppr_percentiles=percentiles,
        disagreement=1.0 - jaccard,
        semantic_top2_mean=semantic_top2_mean,
        semantic_kth_score=float(semantic_kth_score),
        ppr_concentration=ppr_concentration,
        seed_miss_fraction=seed_miss_fraction,
    )


def run_probe(
    dataset: str,
    n_questions: int,
    seed: int,
    *,
    budget_tokens: int | None,
    top_k: int | None,
) -> dict[str, float | int]:
    items = load_hotpotqa(dataset)
    random.Random(seed).shuffle(items)
    items = items[:n_questions]
    observations = []
    for index, item in enumerate(items, 1):
        candidates = generate_candidates(
            item, max_hops=2, use_reranker=True, graph_rel_method="ppr"
        )
        observations.append(
            evaluate_candidates(
                item,
                candidates,
                budget_tokens=budget_tokens,
                top_k=top_k,
                rng_seed=seed + index,
            )
        )
        if index % 20 == 0 or index == len(items):
            print(f"...{index}/{len(items)} questions", flush=True)

    total_gold = sum(o.gold_total for o in observations)
    recoverable_misses = sum(o.recoverable_semantic_misses for o in observations)
    missing_percentiles = [p for o in observations for p in o.missing_gold_ppr_percentiles]
    semantic_incomplete = [o for o in observations if not o.semantic_complete]
    def group_mean(group: list[RescueObservation], field: str) -> float:
        return sum(getattr(o, field) for o in group) / len(group) if group else 0.0

    semantic_complete = [o for o in observations if o.semantic_complete]
    semantic_incomplete = [o for o in observations if not o.semantic_complete]
    rescued = [o for o in semantic_incomplete if o.ppr_rescued_misses > 0]
    result = {
        "n_questions": len(observations),
        "candidate_oracle_recall": sum(o.candidate_gold for o in observations) / total_gold,
        "semantic_recall": sum(o.semantic_gold for o in observations) / total_gold,
        "ppr_recall": sum(o.graph_gold for o in observations) / total_gold,
        "union_recall": sum(o.union_gold for o in observations) / total_gold,
        "semantic_incomplete_questions": len(semantic_incomplete),
        "union_fully_recovers_questions": sum(o.union_complete for o in semantic_incomplete),
        "recoverable_semantic_misses": recoverable_misses,
        "ppr_rescued_misses": sum(o.ppr_rescued_misses for o in observations),
        "ppr_rescue_rate": (
            sum(o.ppr_rescued_misses for o in observations) / recoverable_misses
            if recoverable_misses
            else 0.0
        ),
        "random_rescue_rate": (
            sum(o.random_rescued_misses for o in observations) / recoverable_misses
            if recoverable_misses
            else 0.0
        ),
        "mean_missing_gold_ppr_percentile": (
            sum(missing_percentiles) / len(missing_percentiles) if missing_percentiles else 0.0
        ),
        "proxy_feature_means": {
            group_name: {
                field: group_mean(group, field)
                for field in (
                    "disagreement",
                    "semantic_top2_mean",
                    "semantic_kth_score",
                    "ppr_concentration",
                    "seed_miss_fraction",
                )
            }
            for group_name, group in (
                ("semantic_complete", semantic_complete),
                ("semantic_incomplete", semantic_incomplete),
                ("ppr_rescued", rescued),
            )
        },
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/raw/hotpot_dev_distractor_v1.json")
    parser.add_argument("--n-questions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    operating_point = parser.add_mutually_exclusive_group()
    operating_point.add_argument("--top-k", type=int)
    operating_point.add_argument("--budget-tokens", type=int)
    args = parser.parse_args()
    top_k = args.top_k if args.top_k is not None or args.budget_tokens is not None else 9
    result = run_probe(
        args.dataset,
        args.n_questions,
        args.seed,
        budget_tokens=args.budget_tokens,
        top_k=top_k,
    )
    print(json.dumps(result, indent=2))
