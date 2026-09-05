"""Merge and validate the paired 2WikiMultiHopQA RQ2 B=200 run."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


RUN_NAME = "graft_v1_2wiki_rq2_openrouter_n1000"
ROOT = Path(f"results/{RUN_NAME}_merged")
SHARDS = tuple(
    Path(f"results/{RUN_NAME}_shard{i}/results.jsonl") for i in range(4)
)
RQ1_MERGED = Path(
    "results/graft_v1_2wiki_rq1_openrouter_n1000_merged/results.jsonl"
)
EXPECTED = {
    ("Dense RAG", 200),
    ("Graph 1-hop", 200),
    ("Graph 2-hop", 200),
    ("GRAFT", 200),
}
METHOD_ORDER = {
    ("Dense RAG", 200): 0,
    ("Graph 1-hop", 200): 1,
    ("Graph 2-hop", 200): 2,
    ("GRAFT", 200): 3,
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
CI_METRICS = ("sp_recall", "sp_f1", "em", "f1")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.open() if line.strip()]


def read_rows() -> list[dict]:
    baselines = [row for path in SHARDS for row in load_jsonl(path)]
    graft = [
        row
        for row in load_jsonl(RQ1_MERGED)
        if row["method"] == "GRAFT" and row["budget_tokens"] == 200
    ]
    baseline_qids = {row["qid"] for row in baselines}
    graft_qids = {row["qid"] for row in graft}
    if baseline_qids != graft_qids:
        raise ValueError(
            "RQ2 baseline QIDs do not exactly match the reused RQ1 GRAFT-200 QIDs"
        )
    return baselines + graft


def validate(rows: list[dict]) -> dict:
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_qid[row["qid"]].append(row)
    keys = [(r["qid"], r["method"], r["budget_tokens"]) for r in rows]
    complete = all(
        {(r["method"], r["budget_tokens"]) for r in group} == EXPECTED
        for group in by_qid.values()
    )
    checks = {
        "rows": len(rows),
        "qids": len(by_qid),
        "unique_row_keys": len(set(keys)),
        "rows_per_qid": dict(sorted(Counter(map(len, by_qid.values())).items())),
        "all_method_budget_sets_complete": complete,
        "budget_violations": sum(r["context_tokens"] > 200 for r in rows),
        "blank_predictions": sum(not str(r.get("prediction", "")).strip() for r in rows),
        "config_sha256": sorted({r["config_sha256"] for r in rows}),
        "dataset_sha256": sorted({r["dataset_sha256"] for r in rows}),
        "code_revision": sorted({r["code_revision"] for r in rows}),
        "git_dirty": sorted({r["git_dirty"] for r in rows}),
        "api_backends": sorted({r["api_backend"] for r in rows}),
        "embedding_models": sorted({r["embedding_model"] for r in rows}),
        "reranker_models": sorted({r["reranker_model"] for r in rows}),
        "resolved_llm_models": sorted({r["resolved_llm_model"] for r in rows}),
    }
    failures = []
    if len(rows) != 4_000:
        failures.append("expected 4,000 rows")
    if len(by_qid) != 1_000:
        failures.append("expected 1,000 distinct paired QIDs")
    if len(set(keys)) != len(rows):
        failures.append("duplicate (qid, method, budget) rows")
    if not complete:
        failures.append("one or more QIDs lack the exact four B=200 rows")
    if checks["budget_violations"]:
        failures.append("one or more rows exceed B=200")
    if checks["blank_predictions"]:
        failures.append("one or more predictions are blank")
    if len(checks["config_sha256"]) != 2:
        failures.append("expected separate, internally consistent RQ1 and RQ2 config hashes")
    for field in (
        "dataset_sha256",
        "code_revision",
        "git_dirty",
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


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["budget_tokens"])].append(row)
    output = []
    for key in sorted(groups, key=METHOD_ORDER.get):
        method, budget = key
        group = groups[key]
        item = {
            "benchmark": "2WikiMultiHopQA",
            "method": method,
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
            float(row.get("api_cost_usd") or 0) for row in group
        )
        output.append(item)
    return output


def bootstrap_ci(
    left: list[float], right: list[float], *, seed: int = 0, resamples: int = 10_000
) -> tuple[float, float, float]:
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for start in range(0, resamples, 500):
        stop = min(start + 500, resamples)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        estimates[start:stop] = differences[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(differences.mean()), float(low), float(high)


def paired_cis(rows: list[dict]) -> list[dict]:
    indexed = {
        (row["method"], row["budget_tokens"], row["qid"]): row for row in rows
    }
    qids = sorted({row["qid"] for row in rows})
    output = []
    for comparator in ("Dense RAG", "Graph 1-hop", "Graph 2-hop"):
        for metric in CI_METRICS:
            left = [indexed[("GRAFT", 200, qid)][metric] for qid in qids]
            right = [indexed[(comparator, 200, qid)][metric] for qid in qids]
            delta, low, high = bootstrap_ci(left, right)
            output.append(
                {
                    "reference_method": "GRAFT",
                    "reference_budget": 200,
                    "comparator_method": comparator,
                    "comparator_budget": 200,
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
        "# RQ2 — 2WikiMultiHopQA, matched B=200, N=1,000",
        "",
        "All four methods use the exact same paired QIDs, backend, models, and hard evidence budget.",
        "",
        "| Method | Budget | N | SP Recall ↑ | SP Precision ↑ | SP F1 ↑ | Answer EM ↑ | Answer F1 ↑ | Mean evidence tokens ↓ | Mean prompt tokens ↓ | Mean nodes ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['method']} | {row['budget']} | {row['n']} | "
            f"{fmt(row['sp_recall'])} | {fmt(row['sp_prec'])} | {fmt(row['sp_f1'])} | "
            f"{fmt(row['em'])} | {fmt(row['f1'])} | {fmt(row['context_tokens'], 1)} | "
            f"{fmt(row['prompt_tokens'], 1)} | {fmt(row['num_selected'], 1)} |"
        )
    lines += [
        "",
        "## Paired bootstrap (GRAFT-200 minus comparator-200)",
        "",
        "| Comparator | Metric | Delta | 95% CI |",
        "|---|---|---:|---|",
    ]
    for row in cis:
        if row["metric"] in {"sp_f1", "f1"}:
            lines.append(
                f"| {row['comparator_method']} | {row['metric']} | "
                f"{fmt(row['difference'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] |"
            )
    lines += [
        "",
        "## Validation",
        "",
        f"Passed: **{checks['passed']}**; {checks['rows']} rows; {checks['qids']} paired QIDs; "
        f"{checks['budget_violations']} budget violations; {checks['blank_predictions']} blank predictions.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
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
    print(f"Wrote merged RQ2 artifacts -> {ROOT}")


if __name__ == "__main__":
    main()
