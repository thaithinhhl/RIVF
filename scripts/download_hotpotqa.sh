#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# The original host (curtis.ml.cmu.edu) has been offline since ~May 2025.
# We fetch the equivalent data from the official HF mirror (hotpotqa/hotpot_qa)
# and convert it back to the original nested [title, sentences] / [title, sent_id]
# list-of-pairs schema, so downstream code can rely on the documented official format
# regardless of which mirror supplied the bytes.

python3 - "$@" <<'PY'
import json
import sys
from pathlib import Path

from datasets import load_dataset

Path("data/raw").mkdir(parents=True, exist_ok=True)


def to_official_schema(example: dict) -> dict:
    ctx = example["context"]
    sf = example["supporting_facts"]
    return {
        "_id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "type": example["type"],
        "level": example["level"],
        "supporting_facts": list(zip(sf["title"], sf["sent_id"])),
        "context": list(zip(ctx["title"], ctx["sentences"])),
    }


def dump_split(split: str, out_path: Path) -> None:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    records = [to_official_schema(ds[i]) for i in range(len(ds))]
    out_path.write_text(json.dumps(records))
    print(f"Wrote {len(records)} questions -> {out_path}")


dump_split("validation", Path("data/raw/hotpot_dev_distractor_v1.json"))
if "--with-train" in sys.argv:
    dump_split("train", Path("data/raw/hotpot_train_v1.1.json"))
PY

echo "Done. Files in data/raw/:"
ls -lh data/raw/
