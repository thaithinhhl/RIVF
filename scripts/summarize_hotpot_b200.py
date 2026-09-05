"""Validate, merge, and summarize the HotpotQA Bmax=200 baseline run."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


RUN_NAME = "hotpot_baselines_b200_openrouter_n1000"
BUDGET = 200
ROOT = Path("results") / f"{RUN_NAME}_merged"
SHARDS = tuple(
    Path("results") / f"{RUN_NAME}_shard{index}" / "results.jsonl"
    for index in range(8)
)
GRAFT_RESULTS = Path(
    "results/graft_v2_hotpot_rq1_openrouter_n1000_merged/results.jsonl"
)
METHODS = ("Dense RAG", "Graph 1-hop", "Graph 2-hop")
EXPECTED = {(method, BUDGET) for method in METHODS}
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


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.open() if line.strip()]


def validate(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["qid"]].append(row)
    keys = [(row["qid"], row["method"], row["budget_tokens"]) for row in rows]
    graft_qids = {row["qid"] for row in load_jsonl(GRAFT_RESULTS)}
    run_qids = set(grouped)
    checks = {
        "rows": len(rows),
        "qids": len(run_qids),
        "unique_row_keys": len(set(keys)),
        "rows_per_qid": dict(sorted(Counter(map(len, grouped.values())).items())),
        "all_method_budget_sets_complete": all(
            {(row["method"], row["budget_tokens"]) for row in group} == EXPECTED
            for group in grouped.values()
        ),
        "qid_set_matches_graft": run_qids == graft_qids,
        "qids_missing_from_run": len(graft_qids - run_qids),
        "unexpected_run_qids": len(run_qids - graft_qids),
        "budget_violations": sum(row["context_tokens"] > BUDGET for row in rows),
        "blank_predictions": sum(
            not str(row.get("prediction", "")).strip() for row in rows
        ),
        "rows_with_missing_metrics": sum(
            any(row.get(metric) is None for metric in METRICS) for row in rows
        ),
        "config_sha256": sorted({row["config_sha256"] for row in rows}),
        "dataset_sha256": sorted({row["dataset_sha256"] for row in rows}),
        "api_backends": sorted({row["api_backend"] for row in rows}),
        "embedding_models": sorted({row["embedding_model"] for row in rows}),
        "reranker_models": sorted({row["reranker_model"] for row in rows}),
        "resolved_llm_models": sorted(
            {row["resolved_llm_model"] for row in rows}
        ),
        "reported_answer_cost_usd": sum(
            float(row.get("api_cost_usd") or 0.0) for row in rows
        ),
    }
    failures = []
    if checks["rows"] != 3_000:
        failures.append("expected 3,000 rows")
    if checks["qids"] != 1_000:
        failures.append("expected 1,000 distinct paired QIDs")
    if checks["unique_row_keys"] != checks["rows"]:
        failures.append("duplicate (qid, method, budget) rows")
    if not checks["all_method_budget_sets_complete"]:
        failures.append("one or more QIDs lack the exact three Bmax=200 rows")
    if not checks["qid_set_matches_graft"]:
        failures.append("baseline QID set differs from GRAFT")
    if checks["budget_violations"]:
        failures.append("one or more rows exceed Bmax=200")
    if checks["blank_predictions"]:
        failures.append("one or more predictions are blank")
    if checks["rows_with_missing_metrics"]:
        failures.append("one or more rows have missing metrics")
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


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        item = {
            "benchmark": "HotpotQA",
            "method": method,
            "budget": BUDGET,
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


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict], checks: dict) -> None:
    lines = [
        "# HotpotQA baselines — Bmax=200, N=1,000",
        "",
        "| Method | Bmax | N | SP Recall | SP Precision | SP F1 | Answer EM | Answer F1 | Mean evidence tokens | Mean prompt tokens | Mean nodes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {BUDGET} | {row['n']} | {row['sp_recall']:.4f} | "
            f"{row['sp_prec']:.4f} | {row['sp_f1']:.4f} | {row['em']:.4f} | "
            f"{row['f1']:.4f} | {row['context_tokens']:.1f} | "
            f"{row['prompt_tokens']:.1f} | {row['num_selected']:.1f} |"
        )
    lines += [
        "",
        "## Validation",
        "",
        f"Passed: **{checks['passed']}**; {checks['rows']} rows; "
        f"{checks['qids']} paired QIDs; QID set matches GRAFT: "
        f"**{checks['qid_set_matches_graft']}**; {checks['budget_violations']} "
        f"budget violations; {checks['blank_predictions']} blank predictions; "
        f"{checks['rows_with_missing_metrics']} rows with missing metrics.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = [row for path in SHARDS for row in load_jsonl(path)]
    checks = validate(rows)
    rows.sort(key=lambda row: (row["qid"], METHODS.index(row["method"])))
    aggregates = aggregate(rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (ROOT / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    write_csv(ROOT / "aggregate_metrics.csv", aggregates)
    write_summary(ROOT / "summary.md", aggregates, checks)
    print(json.dumps(checks, indent=2))
    print(f"Wrote merged Bmax={BUDGET} artifacts -> {ROOT}")


if __name__ == "__main__":
    main()
