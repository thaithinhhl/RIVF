#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw

# Official source (Alab-NII/2wikimultihop) hosts data on Dropbox personal
# share links -- live today but more link-rot-prone than an institutional
# host. Using kamelliao's HuggingFace mirror instead: it hosts the original,
# byte-identical JSON files (not restructured into HF's parallel-array
# format), including the id_aliases/evidences fields from the bug-fixed
# data_ids_april7 variant -- verified against a real record before choosing
# this source.
BASE_URL="https://huggingface.co/datasets/kamelliao/2wikimultihopqa/resolve/main/data"

echo "Downloading 2WikiMultiHopQA dev set..."
curl -fL "$BASE_URL/dev.json" -o data/raw/2wikimultihopqa_dev.json

if [[ "${1:-}" == "--with-train" ]]; then
  echo "Downloading 2WikiMultiHopQA train set (large, ~167k questions)..."
  curl -fL "$BASE_URL/train.json" -o data/raw/2wikimultihopqa_train.json
fi

echo "Done. Files in data/raw/:"
ls -lh data/raw/2wikimultihopqa*
