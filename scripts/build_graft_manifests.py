"""Freeze development/evaluation QIDs for GRAFT v1.

All valid QIDs already present in historical result files are assigned to the
development manifest. The remaining valid QIDs form the untouched evaluation
manifest. Raw benchmark files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rivf.data.hotpotqa import load_hotpotqa


DATASETS = {
    "hotpot": Path("data/raw/hotpot_dev_distractor_v1.json"),
    "2wiki": Path("data/raw/2wikimultihopqa_dev.json"),
}


def historical_qids(dataset_path: Path) -> set[str]:
    output: set[str] = set()
    for root in (Path("results"), Path("result_v0.1")):
        if not root.exists():
            continue
        for result_path in root.rglob("results.jsonl"):
            for line in result_path.open():
                row = json.loads(line)
                if Path(row.get("dataset", "")).name == dataset_path.name:
                    output.add(row["qid"])
    return output


def write_manifest(path: Path, *, dataset: Path, role: str, qids: list[str]) -> None:
    payload = {
        "schema_version": 1,
        "method_version": "graft-v1",
        "role": role,
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "hygiene": {
            "deduplicate_passages": True,
            "invalid_gold_policy": "exclude",
        },
        "n_qids": len(qids),
        "qids": qids,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite already-frozen manifests (never use after final runs begin)",
    )
    args = parser.parse_args()
    output_root = Path("data/manifests")
    output_root.mkdir(parents=True, exist_ok=True)
    for name, dataset in DATASETS.items():
        development_path = output_root / f"{name}_graft_v1_development.json"
        evaluation_path = output_root / f"{name}_graft_v1_evaluation.json"
        if not args.force and development_path.exists() and evaluation_path.exists():
            print(f"{name}: manifests already frozen; use --force to rebuild")
            continue
        items = load_hotpotqa(
            dataset,
            deduplicate_passages=True,
            invalid_gold_policy="exclude",
        )
        seen = historical_qids(dataset)
        development = [item.qid for item in items if item.qid in seen]
        evaluation = [item.qid for item in items if item.qid not in seen]
        write_manifest(
            development_path,
            dataset=dataset,
            role="development_seen_in_historical_results",
            qids=development,
        )
        write_manifest(
            evaluation_path,
            dataset=dataset,
            role="evaluation_not_seen_in_historical_results",
            qids=evaluation,
        )
        print(f"{name}: development={len(development)}, evaluation={len(evaluation)}")


if __name__ == "__main__":
    main()
