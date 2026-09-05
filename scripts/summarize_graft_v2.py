"""Validate, merge, and summarize the two GRAFT-v2 N=1,000 runs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


RUNS = {
    "hotpot": {
        "title": "HotpotQA",
        "v1": "graft_v1_hotpot_rq1_openrouter_n1000_merged",
        "v2": "graft_v2_hotpot_rq1_openrouter_n1000",
    },
    "2wiki": {
        "title": "2WikiMultiHopQA",
        "v1": "graft_v1_2wiki_rq1_openrouter_n1000_merged",
        "v2": "graft_v2_2wiki_rq1_openrouter_n1000",
    },
}

METRICS = (
    "sp_recall",
    "sp_prec",
    "sp_f1",
    "em",
    "f1",
    "context_tokens",
    "prompt_tokens",
    "num_selected",
    "output_tokens",
)
CI_METRICS = ("sp_recall", "sp_prec", "sp_f1", "em", "f1")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def validate(rows: list[dict], v1_rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["qid"]].append(row)
    keys = [(row["qid"], row["method"], row["budget_tokens"]) for row in rows]
    expected = {("GRAFT-v2", 600), ("GRAFT-v2", 200)}
    v1_qids = {row["qid"] for row in v1_rows}
    v2_qids = set(grouped)
    checks = {
        "rows": len(rows),
        "qids": len(v2_qids),
        "unique_row_keys": len(set(keys)),
        "rows_per_qid": dict(sorted(Counter(map(len, grouped.values())).items())),
        "all_method_budget_sets_complete": all(
            {(row["method"], row["budget_tokens"]) for row in group} == expected
            for group in grouped.values()
        ),
        "v1_qids": len(v1_qids),
        "qid_set_matches_graft_v1": v2_qids == v1_qids,
        "qids_missing_from_v2": len(v1_qids - v2_qids),
        "unexpected_v2_qids": len(v2_qids - v1_qids),
        "budget_violations": sum(
            row["context_tokens"] > row["budget_tokens"] for row in rows
        ),
        "blank_predictions": sum(
            not str(row.get("prediction", "")).strip() for row in rows
        ),
        "rows_with_missing_metrics": sum(
            any(row.get(metric) is None for metric in METRICS) for row in rows
        ),
        "config_sha256": sorted({row["config_sha256"] for row in rows}),
        "dataset_sha256": sorted({row["dataset_sha256"] for row in rows}),
        "generation_reused": dict(
            sorted(Counter(str(row.get("generation_reused")) for row in rows).items())
        ),
        "paid_answer_calls": sum(
            not row.get("generation_reused", False) for row in rows
        ),
        "reported_answer_cost_usd": sum(
            float(row.get("api_cost_usd") or 0.0) for row in rows
        ),
    }
    failures = []
    if checks["rows"] != 2_000:
        failures.append("expected 2,000 rows")
    if checks["qids"] != 1_000:
        failures.append("expected 1,000 distinct QIDs")
    if checks["unique_row_keys"] != checks["rows"]:
        failures.append("duplicate (qid, method, budget) rows")
    if not checks["all_method_budget_sets_complete"]:
        failures.append("one or more QIDs lack the exact B600 and B200 rows")
    if not checks["qid_set_matches_graft_v1"]:
        failures.append("GRAFT-v2 QID set differs from GRAFT-v1")
    if checks["budget_violations"]:
        failures.append("one or more rows exceed the evidence budget")
    if checks["blank_predictions"]:
        failures.append("one or more predictions are blank")
    if checks["rows_with_missing_metrics"]:
        failures.append("one or more rows have missing metrics")
    if len(checks["config_sha256"]) != 1:
        failures.append("inconsistent config hashes")
    if len(checks["dataset_sha256"]) != 1:
        failures.append("inconsistent dataset hashes")
    checks["passed"] = not failures
    checks["failures"] = failures
    return checks


def aggregate(rows: list[dict], title: str) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["budget_tokens"]].append(row)
    output = []
    for budget in (600, 200):
        group = groups[budget]
        item = {
            "benchmark": title,
            "method": "GRAFT-v2",
            "budget": budget,
            "n": len({row["qid"] for row in group}),
        }
        item.update(
            {
                metric: float(np.mean([float(row[metric]) for row in group]))
                for metric in METRICS
            }
        )
        item["answer_api_cost_usd"] = sum(
            float(row.get("api_cost_usd") or 0.0) for row in group
        )
        output.append(item)
    return output


def paired_cis(rows: list[dict], *, resamples: int = 10_000) -> list[dict]:
    indexed = {
        (row["budget_tokens"], row["qid"]): row
        for row in rows
    }
    qids = sorted({row["qid"] for row in rows})
    rng = np.random.default_rng(0)
    output = []
    for metric in CI_METRICS:
        differences = np.asarray(
            [indexed[(200, qid)][metric] - indexed[(600, qid)][metric] for qid in qids],
            dtype=float,
        )
        estimates = np.empty(resamples, dtype=float)
        for start in range(0, resamples, 500):
            stop = min(start + 500, resamples)
            indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
            estimates[start:stop] = differences[indices].mean(axis=1)
        low, high = np.quantile(estimates, (0.025, 0.975))
        output.append(
            {
                "reference_budget": 200,
                "comparator_budget": 600,
                "metric": metric,
                "n": len(qids),
                "difference": float(differences.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path, title: str, aggregates: list[dict], checks: dict, cis: list[dict]
) -> None:
    lines = [
        f"# GRAFT-v2 — {title}, N=1,000",
        "",
        "| Budget | N | SP Recall | SP Precision | SP F1 | Answer EM | Answer F1 | Evidence tokens | Prompt tokens | Nodes | Output tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['budget']} | {row['n']} | {row['sp_recall']:.4f} | "
            f"{row['sp_prec']:.4f} | {row['sp_f1']:.4f} | {row['em']:.4f} | "
            f"{row['f1']:.4f} | {row['context_tokens']:.1f} | "
            f"{row['prompt_tokens']:.1f} | {row['num_selected']:.1f} | "
            f"{row['output_tokens']:.1f} |"
        )
    lines += [
        "",
        "## Paired B200 - B600 deltas",
        "",
        "| Metric | Delta | 95% bootstrap CI |",
        "|---|---:|---|",
    ]
    for row in cis:
        lines.append(
            f"| {row['metric']} | {row['difference']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |"
        )
    lines += [
        "",
        "## Validation",
        "",
        f"Passed: **{checks['passed']}**; {checks['rows']} rows; {checks['qids']} QIDs; "
        f"QID set matches GRAFT-v1: **{checks['qid_set_matches_graft_v1']}**; "
        f"{checks['budget_violations']} budget violations; "
        f"{checks['blank_predictions']} blank predictions; "
        f"{checks['rows_with_missing_metrics']} rows with missing metrics.",
        "",
        f"Paid answer calls: {checks['paid_answer_calls']}; reported answer cost: "
        f"${checks['reported_answer_cost_usd']:.6f}.",
    ]
    path.write_text("\n".join(lines) + "\n")


def summarize(spec: dict) -> None:
    shard_paths = tuple(
        Path("results") / f"{spec['v2']}_shard{index}" / "results.jsonl"
        for index in range(8)
    )
    for path in shard_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    rows = [row for path in shard_paths for row in read_jsonl(path)]
    v1_rows = read_jsonl(Path("results") / spec["v1"] / "results.jsonl")
    checks = validate(rows, v1_rows)
    if not checks["passed"]:
        raise ValueError("; ".join(checks["failures"]))
    rows.sort(key=lambda row: (row["qid"], -row["budget_tokens"]))
    aggregates = aggregate(rows, spec["title"])
    cis = paired_cis(rows)
    root = Path("results") / f"{spec['v2']}_merged"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (root / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    write_csv(root / "aggregate_metrics.csv", aggregates)
    write_csv(root / "paired_bootstrap_ci.csv", cis)
    write_summary(root / "summary.md", spec["title"], aggregates, checks, cis)
    print(json.dumps({"output": str(root), **checks}, indent=2))


if __name__ == "__main__":
    for run in RUNS.values():
        summarize(run)
