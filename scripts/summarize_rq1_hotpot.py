"""Validate, merge, and summarize a sharded OpenRouter RQ1 run."""

from __future__ import annotations

import csv
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("results/graft_v1_hotpot_rq1_openrouter_n1000_merged")
SHARDS = tuple(
    Path(f"results/graft_v1_hotpot_rq1_openrouter_n1000_shard{i}/results.jsonl")
    for i in range(4)
)
BENCHMARK = "HotpotQA"
TITLE = "HotpotQA"
EXPECTED = {
    ("Dense RAG", 600),
    ("Graph 1-hop", 600),
    ("Graph 2-hop", 600),
    ("GRAFT", 600),
    ("GRAFT", 200),
}
METHOD_ORDER = {
    ("Dense RAG", 600): 0,
    ("Graph 1-hop", 600): 1,
    ("Graph 2-hop", 600): 2,
    ("GRAFT", 600): 3,
    ("GRAFT", 200): 4,
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
)
CI_METRICS = ("sp_recall", "sp_prec", "sp_f1", "em", "f1")


def read_rows() -> list[dict]:
    rows: list[dict] = []
    for path in SHARDS:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(json.loads(line) for line in path.open() if line.strip())
    return rows


def validate(rows: list[dict]) -> dict:
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_qid[row["qid"]].append(row)
    keys = [(r["qid"], r["method"], r["budget_tokens"]) for r in rows]
    method_sets_ok = all(
        {(r["method"], r["budget_tokens"]) for r in group} == EXPECTED
        for group in by_qid.values()
    )
    checks = {
        "rows": len(rows),
        "qids": len(by_qid),
        "unique_row_keys": len(set(keys)),
        "rows_per_qid": dict(sorted(Counter(map(len, by_qid.values())).items())),
        "all_method_budget_sets_complete": method_sets_ok,
        "budget_violations": sum(r["context_tokens"] > r["budget_tokens"] for r in rows),
        "blank_predictions": sum(not str(r.get("prediction", "")).strip() for r in rows),
        "config_sha256": sorted({r["config_sha256"] for r in rows}),
        "dataset_sha256": sorted({r["dataset_sha256"] for r in rows}),
        "api_backends": sorted({r["api_backend"] for r in rows}),
        "embedding_models": sorted({r["embedding_model"] for r in rows}),
        "reranker_models": sorted({r["reranker_model"] for r in rows}),
        "resolved_llm_models": sorted({r["resolved_llm_model"] for r in rows}),
    }
    failures = []
    if len(rows) != 5_000:
        failures.append("expected 5,000 rows")
    if len(by_qid) != 1_000:
        failures.append("expected 1,000 distinct QIDs")
    if len(set(keys)) != len(rows):
        failures.append("duplicate (qid, method, budget) rows")
    if not method_sets_ok:
        failures.append("one or more QIDs lack the exact five method/budget rows")
    if checks["budget_violations"]:
        failures.append("one or more rows exceed the evidence budget")
    if checks["blank_predictions"]:
        failures.append("one or more predictions are blank")
    for field in (
        "config_sha256",
        "dataset_sha256",
        "api_backends",
        "embedding_models",
        "reranker_models",
        "resolved_llm_models",
    ):
        if len(checks[field]) != 1:
            failures.append(f"inconsistent {field}")
    checks["passed"] = not failures
    checks["failures"] = failures
    if failures:
        raise ValueError("; ".join(failures))
    return checks


def mean(group: list[dict], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in group]))


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["budget_tokens"])].append(row)
    output = []
    for key in sorted(groups, key=METHOD_ORDER.get):
        method, budget = key
        group = groups[key]
        item = {
            "benchmark": BENCHMARK,
            "method": method,
            "budget": budget,
            "n": len({row["qid"] for row in group}),
        }
        item.update({metric: mean(group, metric) for metric in METRICS})
        item["answer_api_cost_usd"] = sum(float(row.get("api_cost_usd") or 0) for row in group)
        output.append(item)
    return output


def bootstrap_ci(
    left: list[float], right: list[float], *, seed: int = 0, resamples: int = 10_000
) -> tuple[float, float, float]:
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    # Chunking keeps peak memory small while preserving an exact paired bootstrap.
    chunk = 500
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        estimates[start:stop] = differences[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(differences.mean()), float(low), float(high)


def paired_cis(rows: list[dict]) -> list[dict]:
    indexed = {
        (row["method"], row["budget_tokens"], row["qid"]): row for row in rows
    }
    qids = sorted({row["qid"] for row in rows})
    comparisons = [
        (("GRAFT", 600), ("Dense RAG", 600)),
        (("GRAFT", 600), ("Graph 1-hop", 600)),
        (("GRAFT", 600), ("Graph 2-hop", 600)),
        (("GRAFT", 200), ("Dense RAG", 600)),
        (("GRAFT", 200), ("Graph 1-hop", 600)),
        (("GRAFT", 200), ("Graph 2-hop", 600)),
        (("GRAFT", 200), ("GRAFT", 600)),
    ]
    output = []
    for reference, comparator in comparisons:
        for metric in CI_METRICS:
            left = [indexed[(*reference, qid)][metric] for qid in qids]
            right = [indexed[(*comparator, qid)][metric] for qid in qids]
            delta, low, high = bootstrap_ci(left, right)
            output.append(
                {
                    "reference_method": reference[0],
                    "reference_budget": reference[1],
                    "comparator_method": comparator[0],
                    "comparator_budget": comparator[1],
                    "metric": metric,
                    "n": len(qids),
                    "difference": delta,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_summary(path: Path, aggregates: list[dict], checks: dict, cis: list[dict]) -> None:
    lines = [
        f"# RQ1 — {TITLE}, N=1,000",
        "",
        "All five configurations use the same paired set of 1,000 questions. "
        "The 600-token rows are the same-budget main comparison; GRAFT-200 is the efficiency point.",
        "",
        "| Benchmark | Method | Budget | N | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Mean evidence tokens ↓ | Mean prompt tokens ↓ | Mean nodes ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {BENCHMARK} | {row['method']} | {row['budget']} | {row['n']} | "
            f"{fmt(row['sp_recall'])} | {fmt(row['sp_prec'])} | {fmt(row['sp_f1'])} | "
            f"{fmt(row['em'])} | {fmt(row['f1'])} | {fmt(row['context_tokens'], 1)} | "
            f"{fmt(row['prompt_tokens'], 1)} | {fmt(row['num_selected'], 1)} |"
        )
    key_cis = [row for row in cis if row["metric"] in {"sp_f1", "f1"}]
    lines += [
        "",
        "## Paired bootstrap (10,000 resamples)",
        "",
        "A positive delta favors the reference row.",
        "",
        "| Reference | Comparator | Metric | Delta | 95% CI |",
        "|---|---|---|---:|---|",
    ]
    for row in key_cis:
        lines.append(
            f"| {row['reference_method']}-{row['reference_budget']} | "
            f"{row['comparator_method']}-{row['comparator_budget']} | {row['metric']} | "
            f"{fmt(row['difference'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] |"
        )
    lines += [
        "",
        "## Validation",
        "",
        f"Passed: **{checks['passed']}**; {checks['rows']} rows; {checks['qids']} paired QIDs; "
        f"{checks['budget_violations']} budget violations; {checks['blank_predictions']} blank predictions.",
        "",
        "Full validation metadata is in `validation.json`; all confidence intervals are in "
        "`paired_bootstrap_ci.csv`.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    global ROOT, SHARDS, BENCHMARK, TITLE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=("hotpot", "2wiki"),
        default="hotpot",
        help="Sharded RQ1 run to validate and summarize (default: hotpot)",
    )
    args = parser.parse_args()
    if args.benchmark == "2wiki":
        run_name = "graft_v1_2wiki_rq1_openrouter_n1000"
        ROOT = Path(f"results/{run_name}_merged")
        SHARDS = tuple(
            Path(f"results/{run_name}_shard{i}/results.jsonl")
            for i in range(4)
        )
        BENCHMARK = "2WikiMultiHopQA"
        TITLE = "2WikiMultiHopQA"
    rows = read_rows()
    checks = validate(rows)
    rows.sort(key=lambda row: (row["qid"], METHOD_ORDER[(row["method"], row["budget_tokens"])]))
    aggregates = aggregate(rows)
    cis = paired_cis(rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (ROOT / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    write_csv(ROOT / "aggregate_metrics.csv", aggregates)
    write_csv(ROOT / "paired_bootstrap_ci.csv", cis)
    write_summary(ROOT / "summary.md", aggregates, checks, cis)
    print(json.dumps(checks, indent=2))
    print(f"Wrote merged RQ1 artifacts -> {ROOT}")


if __name__ == "__main__":
    main()
