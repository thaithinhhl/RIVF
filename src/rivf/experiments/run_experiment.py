"""Generic experiment runner: one script, many YAML configs -- not a
separate module per experiment. Each config lists methods (with parameter
sweeps), and this script produces one results/{name}/results.jsonl row per
(question, method, parameter combination).

Usage: python -m rivf.experiments.run_experiment configs/main_comparison.yaml
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterator

import yaml

from rivf.data.hotpotqa import load_hotpotqa
from rivf.data.schema import QAItem, Sentence
from rivf.eval.answer_metrics import answer_metrics
from rivf.eval.efficiency import count_tokens
from rivf.eval.supporting_facts import supporting_fact_metrics
from rivf.generation.llm_client import generate_answer
from rivf.generation.prompts import build_prompt
from rivf.methods.base import Candidate
from rivf.retrieval.adaptive_alpha import estimate_alpha, estimate_alpha_continuous
from rivf.retrieval.embeddings import cosine_sim, embed, embed_batch
from rivf.retrieval.pruning import prune
from rivf.retrieval.scoring import generate_candidates


def _as_list(x) -> list:
    return x if isinstance(x, list) else [x]


def _candidate_to_dict(c: Candidate) -> dict:
    return {
        "title": c.sentence.passage_title,
        "sent_id": c.sentence.sent_id,
        "semantic_score": c.semantic_score,
        "graph_score": c.graph_score,
        "hop_distance": c.hop_distance,
    }


def _run_dense_rag(item: QAItem, cfg: dict) -> Iterator[tuple[dict, list[Candidate], list[Sentence]]]:
    sentences = item.all_sentences()
    q_vec = embed(item.question)
    vecs = embed_batch([s.embedding_text for s in sentences])
    candidates = [
        Candidate(sentence=s, semantic_score=cosine_sim(q_vec, v), embedding=v)
        for s, v in zip(sentences, vecs)
    ]
    for budget in _as_list(cfg.get("budget_tokens", [None])):
        selected = prune(candidates, alpha=1.0, strategy="topk", budget_tokens=budget)
        yield {"budget_tokens": budget, "alpha": None}, candidates, selected


def _run_fixed_hop(item: QAItem, cfg: dict) -> Iterator[tuple[dict, list[Candidate], list[Sentence]]]:
    candidates = generate_candidates(item, max_hops=cfg["hops"])
    for budget in _as_list(cfg.get("budget_tokens", [None])):
        selected = prune(candidates, alpha=1.0, strategy="topk", budget_tokens=budget)
        yield {"budget_tokens": budget, "alpha": None, "hops": cfg["hops"]}, candidates, selected


def _run_selective_graph(item: QAItem, cfg: dict) -> Iterator[tuple[dict, list[Candidate], list[Sentence]]]:
    # The efficiency payoff: candidates (embeddings + graph BFS) are computed
    # ONCE per question here, then every (alpha, budget) combo below is a
    # cheap re-sort/re-slice of the same cached component scores.
    candidates = generate_candidates(
        item,
        max_hops=cfg.get("max_hops", 2),
        use_reranker=cfg.get("use_reranker", False),
        graph_rel_method=cfg.get("graph_rel_method", "hop"),
    )
    strategy = cfg.get("strategy", "topk")
    adaptive = cfg.get("adaptive_alpha", False)  # False | True (binary switch) | "continuous"
    # alpha is a no-op for strategy="random"/"interleave" (neither blends by
    # weight -- random ignores scores, interleave gives both channels equal
    # fixed turns) and for adaptive_alpha (alpha is computed per question,
    # not swept from config), so sweeping it would just repeat identical
    # work under misleading labels.
    alphas = (
        [None] if strategy in ("random", "interleave") or adaptive else _as_list(cfg.get("alpha", [0.5]))
    )
    # Operating point is EITHER a token budget (default) OR a fixed node
    # count -- top_k in config switches the whole sweep to node-count mode.
    top_ks = cfg.get("top_k")
    op_points = _as_list(top_ks) if top_ks is not None else _as_list(cfg.get("budget_tokens", [None]))

    for alpha in alphas:
        for op_point in op_points:
            budget_arg = None if top_ks is not None else op_point
            top_k_arg = op_point if top_ks is not None else None
            run_params = {"budget_tokens": budget_arg, "top_k": top_k_arg}
            if adaptive:
                estimator = estimate_alpha_continuous if adaptive == "continuous" else estimate_alpha
                eff_alpha = estimator(
                    candidates,
                    base_alpha=cfg.get("base_alpha", 0.6),
                    budget_tokens=budget_arg,
                    top_k=top_k_arg,
                )
                selected = prune(
                    candidates, alpha=eff_alpha, strategy=strategy, budget_tokens=budget_arg, top_k=top_k_arg
                )
                yield {**run_params, "alpha": None, "computed_alpha": eff_alpha}, candidates, selected
            else:
                selected = prune(
                    candidates,
                    alpha=alpha if alpha is not None else 0.5,
                    strategy=strategy,
                    budget_tokens=budget_arg,
                    top_k=top_k_arg,
                )
                yield {**run_params, "alpha": alpha}, candidates, selected


_RUNNERS = {
    "dense_rag": _run_dense_rag,
    "fixed_hop": _run_fixed_hop,
    "selective_graph": _run_selective_graph,
}


def _row_key(qid: str, method_name: str, run_params: dict) -> tuple:
    return (
        qid,
        method_name,
        run_params.get("budget_tokens"),
        run_params.get("top_k"),
        run_params.get("alpha"),
    )


def _expected_keys(qid: str, method_cfg: dict) -> set[tuple]:
    """Which (qid, method, budget, top_k, alpha) rows a method_cfg should
    produce, without paying for candidate generation -- used to skip
    already-completed (question, method) pairs entirely on resume, not just
    individual rows."""
    top_ks = method_cfg.get("top_k")
    if top_ks is not None:
        budgets, top_k_list = [None], _as_list(top_ks)
    else:
        budgets, top_k_list = _as_list(method_cfg.get("budget_tokens", [None])), [None]
    if (
        method_cfg["kind"] == "selective_graph"
        and method_cfg.get("strategy", "topk") not in ("random", "interleave")
        and not method_cfg.get("adaptive_alpha", False)
    ):
        alphas = _as_list(method_cfg.get("alpha", [0.5]))
    else:
        alphas = [None]
    return {(qid, method_cfg["name"], b, k, a) for b in budgets for k in top_k_list for a in alphas}


def _load_done_keys(out_path: Path) -> set[tuple]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            row = json.loads(line)
            done.add((row["qid"], row["method"], row.get("budget_tokens"), row.get("top_k"), row.get("alpha")))
    return done


def run(config_path: str) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())
    items = load_hotpotqa(config.get("dataset", "data/raw/hotpot_dev_distractor_v1.json"))
    random.Random(config.get("seed", 0)).shuffle(items)
    items = items[: config["n_questions"]]

    call_llm = config.get("call_llm", False)
    # A dense (alpha x budget) grid search re-slices the SAME candidate list
    # many times per question -- serializing it on every row would mostly be
    # redundant bytes, so wide sweeps can turn it off.
    include_candidates = config.get("include_candidates", True)
    out_path = Path("results") / config["name"] / "results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resumable: a long run calling an external API can be interrupted by a
    # transient network error partway through. Already-written rows are
    # loaded and skipped -- including skipping candidate generation entirely
    # for a (question, method) pair that's already fully done -- so a re-run
    # after a crash picks up where it left off instead of redoing everything.
    done_keys = _load_done_keys(out_path)
    if done_keys:
        print(f"Resuming: {len(done_keys)} rows already done in {out_path}")

    with out_path.open("a") as f:
        for i, item in enumerate(items, 1):
            for method_cfg in config["methods"]:
                if _expected_keys(item.qid, method_cfg) <= done_keys:
                    continue
                runner = _RUNNERS[method_cfg["kind"]]
                for run_params, candidates, selected in runner(item, method_cfg):
                    key = _row_key(item.qid, method_cfg["name"], run_params)
                    if key in done_keys:
                        continue
                    start = time.perf_counter()
                    tokens = sum(count_tokens(s.text) for s in selected)
                    sp = supporting_fact_metrics([s.key for s in selected], item.supporting_facts)
                    row = {
                        "qid": item.qid,
                        "question_type": item.type,
                        "level": item.level,
                        "method": method_cfg["name"],
                        **run_params,
                        "num_selected": len(selected),
                        "context_tokens": tokens,
                        "retrieval_latency_s": time.perf_counter() - start,
                        **sp,
                    }
                    if include_candidates:
                        row["candidates"] = [_candidate_to_dict(c) for c in candidates]
                    if call_llm:
                        prediction = generate_answer(build_prompt(item, selected))
                        row["prediction"] = prediction
                        row.update(answer_metrics(prediction, item.answer))
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    done_keys.add(key)
            if i % 20 == 0 or i == len(items):
                print(f"...{i}/{len(items)} questions done")

    print(f"Wrote results -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    run(parser.parse_args().config)
