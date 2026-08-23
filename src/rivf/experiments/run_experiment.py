"""Generic experiment runner: one script, many YAML configs -- not a
separate module per experiment. Each config lists methods (with parameter
sweeps), and this script produces one results/{name}/results.jsonl row per
(question, method, parameter combination).

Usage: python -m rivf.experiments.run_experiment configs/main_comparison.yaml
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Iterator

import yaml
from openai import RateLimitError

from rivf.data.hotpotqa import load_hotpotqa
from rivf.data.schema import QAItem, Sentence
from rivf.eval.answer_metrics import answer_metrics
from rivf.eval.efficiency import count_tokens
from rivf.eval.supporting_facts import supporting_fact_metrics
from rivf.generation.llm_client import DEFAULT_MODEL, generate_answer
from rivf.generation.prompts import build_prompt
from rivf.methods.base import Candidate
from rivf.retrieval.adaptive_alpha import (
    estimate_alpha,
    estimate_alpha_continuous,
    estimate_alpha_semantic_first,
    semantic_first_signals,
)
from rivf.retrieval.embeddings import cosine_sim, embed, embed_batch
from rivf.retrieval.pruning import graph_rescue_signals, prune
from rivf.retrieval.question_router import (
    BRIDGE_COMPARISON,
    DIRECT_COMPARISON,
    route_question,
)
from rivf.retrieval.reranker import rerank_scores
from rivf.retrieval.scoring import generate_candidates


def _as_list(x) -> list:
    return x if isinstance(x, list) else [x]


def _sample_items(items: list[QAItem], n: int, seed: int, stratified: bool) -> list[QAItem]:
    """Deterministically sample while preserving question-type proportions."""
    if not stratified:
        shuffled = list(items)
        random.Random(seed).shuffle(shuffled)
        return shuffled[:n]
    groups: dict[str, list[QAItem]] = {}
    for item in items:
        groups.setdefault(item.type, []).append(item)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    target = min(n, len(items))
    exact = {key: target * len(group) / len(items) for key, group in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remainder = target - sum(quotas.values())
    order = sorted(groups, key=lambda key: (exact[key] - quotas[key], key), reverse=True)
    for key in order[:remainder]:
        quotas[key] += 1
    sampled = [item for key in sorted(groups) for item in groups[key][:quotas[key]]]
    rng.shuffle(sampled)
    return sampled


def _candidate_to_dict(c: Candidate) -> dict:
    return {
        "title": c.sentence.passage_title,
        "sent_id": c.sentence.sent_id,
        "semantic_score": c.semantic_score,
        "graph_score": c.graph_score,
        "hop_distance": c.hop_distance,
        "path_keys": c.path_keys,
        "neighbor_keys": c.neighbor_keys,
        "conditional_score": c.conditional_score,
        "conditional_anchor_key": c.conditional_anchor_key,
        "text_tokens": count_tokens(c.sentence.text),
    }


RunnerRow = tuple[dict, list[Candidate], list[Sentence], float]


def _generate_fields(prompt: str, model: str, gold_answer: str) -> dict:
    start = time.perf_counter()
    for attempt in range(7):
        try:
            prediction = generate_answer(prompt, model=model)
            break
        except RateLimitError:
            if attempt == 6:
                raise
            time.sleep(min(2 ** (attempt + 1), 30))
    return {
        "llm_latency_s": time.perf_counter() - start,
        "llm_model": model,
        "prompt_tokens": count_tokens(prompt),
        "output_tokens": count_tokens(prediction),
        "prediction": prediction,
        **answer_metrics(prediction, gold_answer),
    }


def _run_dense_rag(item: QAItem, cfg: dict) -> Iterator[RunnerRow]:
    candidate_start = time.perf_counter()
    sentences = item.all_sentences()
    if cfg.get("use_reranker", False):
        scores = rerank_scores(item.question, [s.embedding_text for s in sentences])
        candidates = [Candidate(sentence=s, semantic_score=score) for s, score in zip(sentences, scores)]
    else:
        q_vec = embed(item.question)
        vecs = embed_batch([s.embedding_text for s in sentences])
        candidates = [
            Candidate(sentence=s, semantic_score=cosine_sim(q_vec, v), embedding=v)
            for s, v in zip(sentences, vecs)
        ]
    candidate_latency = time.perf_counter() - candidate_start
    for budget in _as_list(cfg.get("budget_tokens", [None])):
        selection_start = time.perf_counter()
        selected = prune(candidates, alpha=1.0, strategy="topk", budget_tokens=budget)
        latency = candidate_latency + (time.perf_counter() - selection_start)
        yield {"budget_tokens": budget, "alpha": None}, candidates, selected, latency


def _run_fixed_hop(item: QAItem, cfg: dict) -> Iterator[RunnerRow]:
    candidate_start = time.perf_counter()
    candidates = generate_candidates(
        item, max_hops=cfg["hops"], use_reranker=cfg.get("use_reranker", False)
    )
    candidate_latency = time.perf_counter() - candidate_start
    for budget in _as_list(cfg.get("budget_tokens", [None])):
        selection_start = time.perf_counter()
        selected = prune(candidates, alpha=1.0, strategy="topk", budget_tokens=budget)
        latency = candidate_latency + (time.perf_counter() - selection_start)
        yield {"budget_tokens": budget, "alpha": None, "hops": cfg["hops"]}, candidates, selected, latency


def _run_selective_graph(item: QAItem, cfg: dict) -> Iterator[RunnerRow]:
    # The efficiency payoff: candidates (embeddings + graph BFS) are computed
    # ONCE per question here, then every (alpha, budget) combo below is a
    # cheap re-sort/re-slice of the same cached component scores.
    strategy = cfg.get("strategy", "topk")
    routed = strategy in ("routed_topk", "routed_graph_rescue")
    graph_rescue = strategy in ("routed_graph_rescue", "graph_rescue_topk")
    route = route_question(item.question) if routed else None
    bridge_policy = cfg.get("bridge_policy", "single_chain")
    if bridge_policy not in ("single_chain", "dual_chain"):
        raise ValueError("bridge_policy must be 'single_chain' or 'dual_chain'")
    if route == DIRECT_COMPARISON:
        effective_strategy = "coverage_topk"
    elif route == BRIDGE_COMPARISON and bridge_policy == "dual_chain":
        effective_strategy = "chain_set_topk"
    elif graph_rescue:
        effective_strategy = "graph_rescue_topk"
    else:
        effective_strategy = "conditional_topk" if routed else strategy
    conditional_anchor_n = (
        cfg.get("bridge_anchor_n", 2)
        if route == BRIDGE_COMPARISON and bridge_policy == "dual_chain"
        else cfg.get("conditional_anchor_n", 2)
    )

    prefer_mentioned_anchors = (
        cfg.get("prefer_mentioned_anchors", True)
        if route == BRIDGE_COMPARISON and bridge_policy == "dual_chain"
        else False
    )
    cache_key = (
        cfg.get("max_hops", 2), cfg.get("use_reranker", False),
        cfg.get("graph_rel_method", "hop"), cfg.get("conditional_rerank", False),
        conditional_anchor_n, prefer_mentioned_anchors,
        cfg.get("query_weighted_ppr", False), cfg.get("ppr_seed_temperature", 0.20),
    )
    candidate_cache = cfg.get("_candidate_cache")
    cached = candidate_cache.get(cache_key) if candidate_cache is not None else None
    if cached is None:
        candidate_start = time.perf_counter()
        candidates = generate_candidates(
            item,
            max_hops=cfg.get("max_hops", 2),
            use_reranker=cfg.get("use_reranker", False),
            graph_rel_method=cfg.get("graph_rel_method", "hop"),
            conditional_rerank=cfg.get("conditional_rerank", False),
            conditional_anchor_n=conditional_anchor_n,
            prefer_mentioned_anchors=prefer_mentioned_anchors,
            query_weighted_ppr=cfg.get("query_weighted_ppr", False),
            ppr_seed_temperature=cfg.get("ppr_seed_temperature", 0.20),
        )
        candidate_latency = time.perf_counter() - candidate_start
        if candidate_cache is not None:
            candidate_cache[cache_key] = (candidates, candidate_latency)
    else:
        candidates, candidate_latency = cached
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
    rescue_weights = (
        _as_list(cfg.get("graph_rescue_weight", [0.0]))
        if graph_rescue
        else [cfg.get("graph_rescue_weight", 0.0)]
    )

    for alpha in alphas:
        for op_point in op_points:
            for rescue_weight in rescue_weights:
                budget_arg = None if top_ks is not None else op_point
                top_k_arg = op_point if top_ks is not None else None
                run_params = {
                    "budget_tokens": budget_arg,
                    "top_k": top_k_arg,
                    **({"graph_rescue_weight": rescue_weight} if graph_rescue else {}),
                }
                selection_kwargs = {
                    "budget_tokens": budget_arg,
                    "top_k": top_k_arg,
                    "mass_threshold": cfg.get("mass_threshold", 0.65),
                    "mass_temperature": cfg.get("mass_temperature", 0.15),
                    "min_nodes": cfg.get("min_nodes", 4),
                    "path_closure": cfg.get("path_closure", False),
                    "coverage_bonus": cfg.get("coverage_bonus", 0.0),
                    "conditional_weight": cfg.get("conditional_weight", 0.2),
                    "graph_rescue_weight": rescue_weight,
                    "graph_rescue_k": cfg.get("graph_rescue_k", 6),
                    "graph_rescue_gate": cfg.get("graph_rescue_gate", True),
                    "preserve_pairs": cfg.get("preserve_pairs", True),
                }
                rescue_signals = (
                    graph_rescue_signals(candidates, proxy_k=cfg.get("graph_rescue_k", 6))
                    if graph_rescue
                    else {}
                )
                if adaptive:
                    if adaptive == "semantic_first":
                        proxy_k = cfg.get("proxy_k", 6)
                        eff_alpha = estimate_alpha_semantic_first(candidates, proxy_k=proxy_k)
                        proxy_signals = semantic_first_signals(candidates, proxy_k=proxy_k)
                    else:
                        estimator = estimate_alpha_continuous if adaptive == "continuous" else estimate_alpha
                        eff_alpha = estimator(
                            candidates,
                            base_alpha=cfg.get("base_alpha", 0.6),
                            budget_tokens=budget_arg,
                            top_k=top_k_arg,
                        )
                        proxy_signals = {}
                    selection_start = time.perf_counter()
                    selected = prune(
                        candidates, alpha=eff_alpha, strategy=effective_strategy, **selection_kwargs
                    )
                    latency = candidate_latency + (time.perf_counter() - selection_start)
                    yield {
                        **run_params,
                        "alpha": None,
                        "computed_alpha": eff_alpha,
                        **({"computed_route": route} if route is not None else {}),
                        **{f"proxy_{key}": value for key, value in proxy_signals.items()},
                        **{f"graph_rescue_{key}": value for key, value in rescue_signals.items()},
                    }, candidates, selected, latency
                else:
                    selection_start = time.perf_counter()
                    selected = prune(
                        candidates,
                        alpha=alpha if alpha is not None else 0.5,
                        strategy=effective_strategy,
                        **selection_kwargs,
                    )
                    latency = candidate_latency + (time.perf_counter() - selection_start)
                    yield {
                        **run_params,
                        "alpha": alpha,
                        **({"computed_route": route} if route is not None else {}),
                        **{f"graph_rescue_{key}": value for key, value in rescue_signals.items()},
                    }, candidates, selected, latency


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
        run_params.get("graph_rescue_weight"),
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
    rescue_weights = (
        _as_list(method_cfg.get("graph_rescue_weight", [0.0]))
        if method_cfg.get("strategy") in ("routed_graph_rescue", "graph_rescue_topk")
        else [None]
    )
    return {
        (qid, method_cfg["name"], b, k, a, weight)
        for b in budgets
        for k in top_k_list
        for a in alphas
        for weight in rescue_weights
    }


def _load_done_keys(out_path: Path) -> set[tuple]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            row = json.loads(line)
            done.add(
                (
                    row["qid"],
                    row["method"],
                    row.get("budget_tokens"),
                    row.get("top_k"),
                    row.get("alpha"),
                    row.get("graph_rescue_weight"),
                )
            )
    return done


def run(config_path: str) -> Path:
    config_text = Path(config_path).read_text()
    config = yaml.safe_load(config_text)
    config_sha256 = hashlib.sha256(config_text.encode()).hexdigest()
    dataset_path = config.get("dataset", "data/raw/hotpot_dev_distractor_v1.json")
    sample_seed = config.get("seed", 0)
    items = load_hotpotqa(dataset_path)
    items = _sample_items(
        items,
        config["n_questions"],
        sample_seed,
        stratified=config.get("stratified_sampling", False),
    )

    call_llm = config.get("call_llm", False) and os.environ.get("RIVF_SKIP_LLM") != "1"
    llm_model = config.get("llm_model", DEFAULT_MODEL)
    # A dense (alpha x budget) grid search re-slices the SAME candidate list
    # many times per question -- serializing it on every row would mostly be
    # redundant bytes, so wide sweeps can turn it off.
    include_candidates = config.get("include_candidates", True)
    output_root = os.environ.get("RIVF_OUTPUT_ROOT", config.get("output_root", "results"))
    output_name = os.environ.get("RIVF_OUTPUT_NAME", config["name"])
    out_path = Path(output_root) / output_name / "results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resumable: a long run calling an external API can be interrupted by a
    # transient network error partway through. Already-written rows are
    # loaded and skipped -- including skipping candidate generation entirely
    # for a (question, method) pair that's already fully done -- so a re-run
    # after a crash picks up where it left off instead of redoing everything.
    done_keys = _load_done_keys(out_path)
    if out_path.exists():
        prior_hashes = {
            row["config_sha256"]
            for row in (json.loads(line) for line in out_path.open())
            if row.get("config_sha256")
        }
        if prior_hashes and prior_hashes != {config_sha256}:
            raise ValueError(
                f"Config changed for existing result file {out_path}; use a new experiment name"
            )
    if done_keys:
        print(f"Resuming: {len(done_keys)} rows already done in {out_path}")

    with out_path.open("a") as f:
        for i, item in enumerate(items, 1):
            pending: list[tuple[tuple, dict, str | None]] = []
            candidate_cache: dict[tuple, tuple[list[Candidate], float]] = {}
            for raw_method_cfg in config["methods"]:
                method_cfg = {**raw_method_cfg, "_candidate_cache": candidate_cache}
                if _expected_keys(item.qid, method_cfg) <= done_keys:
                    continue
                runner = _RUNNERS[method_cfg["kind"]]
                for run_params, candidates, selected, retrieval_latency_s in runner(item, method_cfg):
                    key = _row_key(item.qid, method_cfg["name"], run_params)
                    if key in done_keys:
                        continue
                    tokens = sum(count_tokens(s.text) for s in selected)
                    sp = supporting_fact_metrics([s.key for s in selected], item.supporting_facts)
                    row = {
                        "qid": item.qid,
                        "question_type": item.type,
                        "level": item.level,
                        "method": method_cfg["name"],
                        "dataset": dataset_path,
                        "sample_seed": sample_seed,
                        "config_sha256": config_sha256,
                        "use_reranker": method_cfg.get("use_reranker", False),
                        **run_params,
                        "num_selected": len(selected),
                        "context_tokens": tokens,
                        "retrieval_latency_s": retrieval_latency_s,
                        "selected_keys": [list(s.key) for s in selected],
                        **sp,
                    }
                    if include_candidates:
                        row["candidates"] = [_candidate_to_dict(c) for c in candidates]
                    if call_llm:
                        chain_aware_routes = method_cfg.get("chain_aware_routes")
                        chain_aware_prompt = method_cfg.get("chain_aware_prompt", False)
                        if chain_aware_routes is not None:
                            chain_aware_prompt = route_question(item.question) in chain_aware_routes
                        prompt = build_prompt(
                            item,
                            selected,
                            chain_aware=chain_aware_prompt,
                        )
                    else:
                        prompt = None
                    pending.append((key, row, prompt))
            if call_llm and pending:
                configured_workers = int(
                    os.environ.get("RIVF_LLM_CONCURRENCY", config.get("llm_concurrency", 6))
                )
                workers = min(configured_workers, len(pending))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    fields = executor.map(
                        lambda entry: _generate_fields(entry[2], llm_model, item.answer),
                        pending,
                    )
                    for (_, row, _), generated in zip(pending, fields):
                        row.update(generated)
            for key, row, _ in pending:
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
