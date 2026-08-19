"""Go/no-go checkpoint: can the entity+title graph actually connect gold
evidence within 1-2 hops of real seed retrieval? No LLM, no scoring, no
pruning -- this only tests whether the graph-construction idea is even
viable before any method/baseline code gets written.

Usage: python scripts/oracle_probe.py [n_questions] [seed] [dataset_path]
Default dataset_path is HotpotQA's dev distractor set; pass
data/raw/2wikimultihopqa_dev.json to run the same check on the secondary
benchmark (same loader works -- record shape is identical, see
src/rivf/data/twowiki.py).
"""

import random
import sys

from rivf.data.hotpotqa import load_hotpotqa
from rivf.eval.supporting_facts import supporting_fact_metrics
from rivf.graph.build import bfs_hop_distances, build_graph
from rivf.retrieval.seed import seed_retrieve


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    dataset_path = sys.argv[3] if len(sys.argv) > 3 else "data/raw/hotpot_dev_distractor_v1.json"

    items = load_hotpotqa(dataset_path)
    random.Random(seed).shuffle(items)
    items = items[:n]

    recalls = {1: [], 2: []}
    sizes = {1: [], 2: []}
    bridge_full_recall_at_2 = []

    for i, item in enumerate(items, 1):
        g = build_graph(item)
        seed_sentences = seed_retrieve(item)
        seed_keys = [s.key for s in seed_sentences]
        dist = bfs_hop_distances(g, seed_keys)

        for hop in (1, 2):
            candidate_keys = [k for k, d in dist.items() if d <= hop]
            r = supporting_fact_metrics(candidate_keys, item.supporting_facts)["sp_recall"]
            recalls[hop].append(r)
            sizes[hop].append(len(candidate_keys))
            # "bridge" (HotpotQA) / "bridge_comparison" (2WikiMultiHopQA) are
            # each dataset's multi-hop-via-shared-entity question type.
            if hop == 2 and item.type in ("bridge", "bridge_comparison"):
                bridge_full_recall_at_2.append(r >= 0.999)

        if i % 50 == 0:
            print(f"...{i}/{len(items)} processed", file=sys.stderr)

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\n=== Oracle reachability probe (n={len(items)}, seed_n={5}) ===")
    for hop in (1, 2):
        print(
            f"hop={hop}: mean gold-sentence recall = {avg(recalls[hop]):.3f}, "
            f"mean candidate-set size = {avg(sizes[hop]):.1f}"
        )
    print(
        f"bridge questions with FULL gold recall at 2-hop: "
        f"{avg(bridge_full_recall_at_2):.3f} "
        f"({sum(bridge_full_recall_at_2)}/{len(bridge_full_recall_at_2)})"
    )


if __name__ == "__main__":
    main()
