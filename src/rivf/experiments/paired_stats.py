"""Paired bootstrap confidence intervals for experiment JSONL files.

Usage:
    python -m rivf.experiments.paired_stats \
        results/fair_reranker_hotpot_200/results.jsonl \
        --reference ours_ce
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_METRICS = ("sp_recall", "sp_f1", "em", "f1", "context_tokens")


def paired_bootstrap(
    reference: pd.Series,
    comparator: pd.Series,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean paired difference and percentile 95% bootstrap interval."""
    paired = pd.concat([reference, comparator], axis=1, join="inner").dropna()
    if paired.empty:
        raise ValueError("no paired observations")
    differences = paired.iloc[:, 0].to_numpy(float) - paired.iloc[:, 1].to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = differences[rng.integers(0, len(differences), size=(n_resamples, len(differences)))]
    low, high = np.quantile(sampled.mean(axis=1), [0.025, 0.975])
    return float(differences.mean()), float(low), float(high)


def comparison_table(
    results_path: str | Path,
    reference_method: str,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(results_path).open()]
    data = pd.DataFrame(rows)
    if data.duplicated(["qid", "method"]).any():
        raise ValueError("expected one row per (qid, method)")
    indexed = data.set_index(["method", "qid"])
    methods = sorted(set(data["method"]) - {reference_method})
    output = []
    for comparator in methods:
        for metric in metrics:
            reference = indexed.loc[reference_method][metric]
            baseline = indexed.loc[comparator][metric]
            diff, low, high = paired_bootstrap(
                reference, baseline, n_resamples=n_resamples, seed=seed
            )
            output.append(
                {
                    "reference": reference_method,
                    "comparator": comparator,
                    "metric": metric,
                    "n": len(reference.index.intersection(baseline.index)),
                    "reference_mean": reference.mean(),
                    "comparator_mean": baseline.mean(),
                    "difference": diff,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    table = comparison_table(
        args.results,
        args.reference,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    print(table.round(4).to_string(index=False))
