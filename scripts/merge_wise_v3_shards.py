"""Validate and merge the three WISE-v3 N=1,000 evaluation shards."""

from __future__ import annotations

import json
import math
from pathlib import Path
from collections import Counter, defaultdict
import statistics

from rivf.data.hotpotqa import load_hotpotqa
from rivf.experiments.run_experiment import _filter_qid_manifest, _sample_items


ROOT = Path("results/wise_v3_2wiki_answer_b200_openrouter_n1000")
SHARDS = tuple(
    Path(
        f"results/wise_v3_2wiki_answer_b200_openrouter_n1000_shard{index}"
        "/results.jsonl"
    )
    for index in range(3)
)
MANIFEST = Path("data/manifests/2wiki_graft_v1_evaluation.json")
DATASET = Path("data/raw/2wikimultihopqa_dev.json")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    items = load_hotpotqa(
        str(DATASET), deduplicate_passages=True, invalid_gold_policy="exclude"
    )
    items = _filter_qid_manifest(
        items, str(MANIFEST), manifest["dataset_sha256"]
    )
    expected_qids = [
        item.qid
        for item in _sample_items(items, 1000, seed=0, stratified=True)
    ]
    rows = [
        json.loads(line)
        for path in SHARDS
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    by_qid: dict[str, dict] = {}
    duplicates = []
    for row in rows:
        qid = row["qid"]
        if qid in by_qid:
            duplicates.append(qid)
        by_qid[qid] = row

    missing = sorted(set(expected_qids) - set(by_qid))
    unexpected = sorted(set(by_qid) - set(expected_qids))
    config_hashes = sorted({row.get("config_sha256") for row in rows})
    methods = sorted({row.get("method") for row in rows})
    budgets = sorted({row.get("budget_tokens") for row in rows})
    failures = []
    if len(rows) != 1000:
        failures.append(f"expected 1000 rows, found {len(rows)}")
    if duplicates:
        failures.append(f"duplicate QIDs: {duplicates[:5]}")
    if missing:
        failures.append(f"missing QIDs: {missing[:5]}")
    if unexpected:
        failures.append(f"unexpected QIDs: {unexpected[:5]}")
    if len(config_hashes) != 1:
        failures.append(f"inconsistent config hashes: {config_hashes}")
    if methods != ["WISE-v3"]:
        failures.append(f"unexpected methods: {methods}")
    if budgets != [200]:
        failures.append(f"unexpected budgets: {budgets}")
    if failures:
        raise ValueError("; ".join(failures))

    ordered = [by_qid[qid] for qid in expected_qids]
    output = ROOT / "results.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered)
    )
    checks = {
        "rows": len(ordered),
        "unique_qids": len(by_qid),
        "missing_qids": missing,
        "unexpected_qids": unexpected,
        "duplicate_qids": duplicates,
        "config_sha256": config_hashes[0],
        "methods": methods,
        "budgets": budgets,
        "passed": True,
    }
    (ROOT / "merge_validation.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n"
    )
    shard_metadata = [
        json.loads((path.parent / "run_metadata.json").read_text())
        for path in SHARDS
    ]
    metadata = {
        "config": shard_metadata[0]["config"],
        "config_sha256": config_hashes[0],
        "code_revision": shard_metadata[0]["code_revision"],
        "git_dirty": any(value["git_dirty"] for value in shard_metadata),
        "completed_rows": len(ordered),
        "merged_shards": len(SHARDS),
        "shard_rows": [value["completed_rows"] for value in shard_metadata],
        "embedding_backend": shard_metadata[0]["embedding_backend"],
        "embedding_model": shard_metadata[0]["embedding_model"],
        "embedding_api_usage": {
            key: sum(value["embedding_api_usage"][key] for value in shard_metadata)
            for key in ("calls", "prompt_tokens", "total_tokens")
        },
        "reranker_backend": shard_metadata[0]["reranker_backend"],
        "reranker_model": shard_metadata[0]["reranker_model"],
        "reranker_api_usage": {
            key: sum(value["reranker_api_usage"][key] for value in shard_metadata)
            for key in ("calls", "search_units", "total_tokens")
        },
        "llm_backend": shard_metadata[0]["llm_backend"],
        "llm_model": shard_metadata[0]["llm_model"],
    }
    (ROOT / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    successful = [row for row in ordered if not row.get("wise_error")]

    def averages(group: list[dict]) -> dict:
        metrics = (
            "em", "f1", "sp_em", "sp_f1", "sp_prec", "sp_recall",
            "context_tokens", "constructed_prompt_tokens",
            "retrieval_latency_s", "llm_latency_s",
        )
        return {
            metric: sum(float(row.get(metric) or 0) for row in group) / len(group)
            for metric in metrics
        }

    serialized = sorted(row["wise_serialized_tokens"] for row in successful)
    percentile_index = math.ceil(0.95 * len(serialized)) - 1
    item_types = {item.qid: item.type for item in items}
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in ordered:
        by_type[item_types[row["qid"]]].append(row)
    summary = {
        "n": len(ordered),
        "certified": len(successful),
        "certificate_rate": len(successful) / len(ordered),
        "errors": dict(sorted(Counter(
            row["wise_error"]["code"]
            for row in ordered
            if row.get("wise_error")
        ).items())),
        "all_sample": averages(ordered),
        "certified_only": averages(successful),
        "serialized_evidence_tokens_certified": {
            "mean": statistics.mean(serialized),
            "median": statistics.median(serialized),
            "p95": serialized[percentile_index],
            "max": max(serialized),
            "over_budget": sum(value > 200 for value in serialized),
        },
        "by_type": {
            kind: {
                "n": len(group),
                "certified": sum(not row.get("wise_error") for row in group),
                **averages(group),
            }
            for kind, group in sorted(by_type.items())
        },
    }
    (ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
