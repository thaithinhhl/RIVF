import json
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
CANONICAL_CONFIGS = (
    "final_hotpot_full_retrieval.yaml",
    "final_2wiki_full_retrieval.yaml",
    "graft_v1_hotpot_answer.yaml",
    "graft_v1_2wiki_answer.yaml",
    "graft_v1_hotpot_budget.yaml",
    "graft_v1_2wiki_budget.yaml",
    "graft_v1_hotpot_ablation.yaml",
    "graft_v1_2wiki_ablation.yaml",
)


def test_canonical_graft_constants_are_frozen():
    for filename in CANONICAL_CONFIGS:
        config = yaml.safe_load((ROOT / "configs" / filename).read_text())
        graft_methods = [
            method
            for method in config["methods"]
            if method["name"] in {"GRAFT", "Full GRAFT"}
        ]
        assert len(graft_methods) == 1
        method = graft_methods[0]
        assert method["max_hops"] == 2
        assert method["conditional_anchor_n"] == 2
        assert method["conditional_link_top_l"] == 2
        assert method["conditional_weight"] == 0.30
        assert method["alpha"] == 1.0
        assert method["graph_edge_types"] == [
            "same_passage",
            "entity_overlap",
            "title_mention",
        ]


def test_development_and_evaluation_manifests_are_disjoint():
    for dataset in ("hotpot", "2wiki"):
        development = json.loads(
            (ROOT / "data" / "manifests" / f"{dataset}_graft_v1_development.json").read_text()
        )
        evaluation = json.loads(
            (ROOT / "data" / "manifests" / f"{dataset}_graft_v1_evaluation.json").read_text()
        )
        assert set(development["qids"]).isdisjoint(evaluation["qids"])
        assert development["n_qids"] == len(development["qids"])
        assert evaluation["n_qids"] == len(evaluation["qids"])


def test_full_configs_use_every_evaluation_qid():
    for filename in ("final_hotpot_full_retrieval.yaml", "final_2wiki_full_retrieval.yaml"):
        config = yaml.safe_load((ROOT / "configs" / filename).read_text())
        manifest = json.loads((ROOT / config["qid_manifest"]).read_text())
        assert config["n_questions"] == manifest["n_qids"]
        assert config["deduplicate_passages"] is True
        assert config["invalid_gold_policy"] == "exclude"
