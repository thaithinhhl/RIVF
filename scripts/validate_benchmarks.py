"""Preflight the two canonical GRAFT benchmarks before an experiment run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rivf.data.hotpotqa import audit_hotpot_style_records, load_hotpotqa


DEFAULT_DATASETS = (
    Path("data/raw/hotpot_dev_distractor_v1.json"),
    Path("data/raw/2wikimultihopqa_dev.json"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", type=Path, default=DEFAULT_DATASETS)
    args = parser.parse_args()

    for path in args.datasets:
        raw = json.loads(path.read_text())
        audit = audit_hotpot_style_records(raw)
        canonical = load_hotpotqa(
            path,
            deduplicate_passages=True,
            invalid_gold_policy="exclude",
        )
        print(f"{path}")
        print(f"  sha256: {sha256(path)}")
        for key, value in audit.items():
            print(f"  raw_{key}: {value}")
        print(f"  canonical_records: {len(canonical)}")

        if audit["duplicate_qids"] or audit["conflicting_passage_records"]:
            raise SystemExit(
                f"fatal benchmark ambiguity in {path}; resolve before running experiments"
            )


if __name__ == "__main__":
    main()
