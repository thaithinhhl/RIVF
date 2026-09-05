"""Generic experiment runner: one script, many YAML configs -- not a
separate module per experiment. Each config lists methods (with parameter
sweeps), and this script produces one results/{name}/results.jsonl row per
(question, method, parameter combination).

Usage: python -m rivf.experiments.run_experiment configs/main_comparison.yaml
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
import random
import subprocess
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
from rivf.generation.llm_client import (
    DEFAULT_BACKEND as DEFAULT_LLM_BACKEND,
    DEFAULT_MODEL,
    generate_answer_with_metadata,
)
from rivf.generation.prompts import build_prompt
from rivf.graph.build import ALL_EDGE_TYPES
from rivf.methods.base import Candidate
from rivf.retrieval.adaptive_alpha import (
    estimate_alpha,
    estimate_alpha_continuous,
    estimate_alpha_semantic_first,
    semantic_first_signals,
)
from rivf.retrieval.embeddings import (
    backend_name as embedding_backend_name,
    configure as configure_embeddings,
    cosine_sim,
    embed,
    embed_batch,
    model_name as embedding_model_name,
    usage as embedding_usage,
)
from rivf.retrieval.pruning import VALID_STRATEGIES, graph_rescue_signals, prune
from rivf.retrieval.question_router import (
    BRIDGE_COMPARISON,
    DIRECT_COMPARISON,
    route_question,
    route_question_v2,
)
from rivf.retrieval.reranker import (
    backend_name as reranker_backend_name,
    configure as configure_reranker,
    model_name as reranker_model_name,
    rerank_scores,
    usage as reranker_usage,
)
from rivf.retrieval.scoring import generate_candidates, question_mentioned_titles
from rivf.retrieval.wise_v3 import (
    WiseV3Error,
    build_typed_evidence_graph,
    induce_reasoning_plan,
    propose_candidates,
    solve_minimum_cost_witness,
    verify_claim_assignments,
    verify_dependencies,
)


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
        "conditional_gain": c.conditional_gain,
        "conditional_anchor_key": c.conditional_anchor_key,
        "conditional_validated": c.conditional_validated,
        "text_tokens": count_tokens(c.sentence.text),
    }


RunnerRow = tuple[dict, list[Candidate], list[Sentence], float]


def _generate_fields(
    prompt: str,
    model: str,
    gold_answer: str,
    backend: str = DEFAULT_LLM_BACKEND,
) -> dict:
    start = time.perf_counter()
    for attempt in range(7):
        try:
            generation = generate_answer_with_metadata(
                prompt, model=model, backend=backend
            )
            break
        except RateLimitError:
            if attempt == 6:
                raise
            time.sleep(min(2 ** (attempt + 1), 30))
    prediction = generation["prediction"]
    return {
        "llm_latency_s": time.perf_counter() - start,
        "llm_model": model,
        "prompt_tokens": count_tokens(prompt),
        "output_tokens": count_tokens(prediction),
        "prediction": prediction,
        **{key: value for key, value in generation.items() if key != "prediction"},
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
        item,
        max_hops=cfg["hops"],
        use_reranker=cfg.get("use_reranker", False),
        graph_edge_types=set(cfg["graph_edge_types"])
        if cfg.get("graph_edge_types") is not None
        else None,
    )
    candidate_latency = time.perf_counter() - candidate_start
    for budget in _as_list(cfg.get("budget_tokens", [None])):
        selection_start = time.perf_counter()
        selected = prune(candidates, alpha=1.0, strategy="topk", budget_tokens=budget)
        latency = candidate_latency + (time.perf_counter() - selection_start)
        yield {"budget_tokens": budget, "alpha": None, "hops": cfg["hops"]}, candidates, selected, latency


def _validate_wise_v3_config(
    cfg: dict, *, default_planner_backend: str, default_planner_model: str
) -> None:
    forbidden = {
        "strategy",
        "alpha",
        "adaptive_alpha",
        "graph_rescue_weight",
        "witness_rescue_max_units",
        "witness_fallback_min_nodes",
        "bridge_witness_fallback_min_nodes",
    }
    present = sorted(forbidden & set(cfg))
    if present:
        raise WiseV3Error(
            "configuration_invalid",
            "alternative-policy parameters are forbidden",
            parameters=present,
        )
    if cfg.get("use_reranker") is not True:
        raise WiseV3Error(
            "configuration_invalid", "WISE-v3 requires use_reranker: true"
        )
    budgets = _as_list(cfg.get("budget_tokens"))
    if not budgets or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in budgets
    ):
        raise WiseV3Error(
            "configuration_invalid", "budget_tokens must be positive integers"
        )
    for parameter, default in (
        ("seed_n", 5),
        ("claim_top_k", 5),
        ("claim_prefilter_k", 8),
        ("dependency_path_top_k", 3),
        ("max_search_states", 100_000),
    ):
        value = cfg.get(parameter, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise WiseV3Error(
                "configuration_invalid", f"{parameter} must be a positive integer"
            )
    if cfg.get("claim_prefilter_k", 8) < cfg.get("claim_top_k", 5):
        raise WiseV3Error(
            "configuration_invalid", "claim_prefilter_k must be at least claim_top_k"
        )
    for parameter, default in (("max_hops", 2), ("max_path_hops", 3)):
        value = cfg.get(parameter, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WiseV3Error(
                "configuration_invalid", f"{parameter} must be a non-negative integer"
            )
    for parameter, default in (
        ("claim_support_threshold", 0.7),
        ("dependency_threshold", 0.7),
        ("witness_sufficiency_threshold", 0.8),
    ):
        value = cfg.get(parameter, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise WiseV3Error(
                "configuration_invalid", f"{parameter} must be numeric in [0,1]"
            )
    temperature = cfg.get("confidence_temperature", 1.0)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or float(temperature) <= 0.0
    ):
        raise WiseV3Error(
            "configuration_invalid", "confidence_temperature must be positive"
        )
    planner_backend = cfg.get("planner_backend", default_planner_backend)
    planner_model = cfg.get("planner_model", default_planner_model)
    if planner_backend not in {"openai", "openrouter"}:
        raise WiseV3Error(
            "configuration_invalid", "planner_backend must be openai or openrouter"
        )
    if not isinstance(planner_model, str) or not planner_model.strip():
        raise WiseV3Error(
            "configuration_invalid", "planner_model must be a non-empty string"
        )
    verifier_backend = cfg.get("verifier_backend", planner_backend)
    verifier_model = cfg.get("verifier_model", planner_model)
    if verifier_backend not in {"openai", "openrouter"}:
        raise WiseV3Error(
            "configuration_invalid", "verifier_backend must be openai or openrouter"
        )
    if not isinstance(verifier_model, str) or not verifier_model.strip():
        raise WiseV3Error(
            "configuration_invalid", "verifier_model must be a non-empty string"
        )


def _run_wise_v3_strict(item: QAItem, cfg: dict) -> Iterator[RunnerRow]:
    """Run one WISE plan/proposal/verification pass, then exact budget solves."""
    _validate_wise_v3_config(
        cfg,
        default_planner_backend=cfg["_default_planner_backend"],
        default_planner_model=cfg["_default_planner_model"],
    )
    verification_start = time.perf_counter()
    planner_model = cfg.get("planner_model", cfg["_default_planner_model"])
    planner_backend = cfg.get("planner_backend", cfg["_default_planner_backend"])
    verifier_model = cfg.get("verifier_model", planner_model)
    verifier_backend = cfg.get("verifier_backend", planner_backend)
    plan, planner_metadata = induce_reasoning_plan(
        item,
        model=planner_model,
        backend=planner_backend,
    )
    graph = build_typed_evidence_graph(item)
    candidates = propose_candidates(
        item,
        plan,
        graph,
        seed_n=int(cfg.get("seed_n", 5)),
        max_hops=int(cfg.get("max_hops", 2)),
    )
    assignments, claim_metadata = verify_claim_assignments(
        plan,
        candidates,
        threshold=float(cfg.get("claim_support_threshold", 0.7)),
        top_k=int(cfg.get("claim_top_k", 5)),
        prefilter_k=int(cfg.get("claim_prefilter_k", 8)),
        model=verifier_model,
        backend=verifier_backend,
        confidence_temperature=float(cfg.get("confidence_temperature", 1.0)),
    )
    dependencies, dependency_metadata = verify_dependencies(
        plan,
        candidates,
        graph,
        assignments,
        threshold=float(cfg.get("dependency_threshold", 0.7)),
        max_path_hops=int(cfg.get("max_path_hops", 3)),
        path_top_k=int(cfg.get("dependency_path_top_k", 3)),
        model=verifier_model,
        backend=verifier_backend,
        confidence_temperature=float(cfg.get("confidence_temperature", 1.0)),
    )
    verification_latency = time.perf_counter() - verification_start

    for budget in _as_list(cfg["budget_tokens"]):
        selection_start = time.perf_counter()
        witness = solve_minimum_cost_witness(
            plan,
            candidates,
            assignments,
            dependencies,
            budget_tokens=int(budget),
            max_search_states=int(cfg.get("max_search_states", 100_000)),
            sufficiency_threshold=float(
                cfg.get("witness_sufficiency_threshold", 0.8)
            ),
        )
        witness = replace(
            witness,
            planner_metadata={
                "planner": planner_metadata,
                "claim_verifier": claim_metadata,
                "dependency_verifier": dependency_metadata,
            },
        )
        latency = verification_latency + (time.perf_counter() - selection_start)
        yield {
            "budget_tokens": int(budget),
            "alpha": None,
            **witness.diagnostics(),
        }, candidates, list(witness.selected), latency


def _run_wise_v3(item: QAItem, cfg: dict) -> Iterator[RunnerRow]:
    """Return explicit zero-score error rows; never substitute another policy."""
    start = time.perf_counter()
    try:
        yield from _run_wise_v3_strict(item, cfg)
    except WiseV3Error as exc:
        latency = time.perf_counter() - start
        for budget in _as_list(cfg.get("budget_tokens")):
            yield {
                "budget_tokens": budget,
                "alpha": None,
                "wise_failed": True,
                "wise_error": exc.to_dict(),
                "wise_certified": False,
                "wise_irreducible": False,
                "wise_operation_complete": False,
            }, [], [], latency


def _run_selective_graph(item: QAItem, cfg: dict) -> Iterator[RunnerRow]:
    # The efficiency payoff: candidates (embeddings + graph BFS) are computed
    # ONCE per question here, then every (alpha, budget) combo below is a
    # cheap re-sort/re-slice of the same cached component scores.
    strategy = cfg.get("strategy", "topk")
    graft_v2 = strategy == "graft_v2"
    routed = strategy in ("graft", "graft_v2", "routed_topk", "routed_graph_rescue")
    graph_rescue = strategy in ("routed_graph_rescue", "graph_rescue_topk")
    if graft_v2:
        route = route_question_v2(item.question)
    elif routed:
        route = route_question(item.question)
    else:
        route = None
    bridge_policy = cfg.get("bridge_policy", "single_chain")
    if bridge_policy not in ("single_chain", "dual_chain"):
        raise ValueError("bridge_policy must be 'single_chain' or 'dual_chain'")
    if route == DIRECT_COMPARISON:
        effective_strategy = "entity_witness" if graft_v2 else "coverage_topk"
    elif graft_v2:
        effective_strategy = "graft_v2_witness"
    elif strategy == "graft":
        effective_strategy = "graft_pair_topk"
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

    prefer_mentioned_anchors = cfg.get("prefer_mentioned_anchors", True) if (
        route == BRIDGE_COMPARISON
        and (bridge_policy == "dual_chain" or graft_v2)
    ) else False
    conditional_rerank = cfg.get("conditional_rerank", False)
    if strategy in ("graft", "graft_v2") and route == DIRECT_COMPARISON:
        conditional_rerank = False
    direct_all_sentence_pool = (
        graft_v2
        and route == DIRECT_COMPARISON
        and cfg.get("direct_all_sentence_pool", True)
    )
    configured_edge_types = cfg.get("graph_edge_types")
    graph_edge_types = (
        tuple(configured_edge_types) if configured_edge_types is not None else None
    )
    cache_key = (
        cfg.get("max_hops", 2), cfg.get("use_reranker", False),
        cfg.get("graph_rel_method", "hop"), conditional_rerank,
        conditional_anchor_n, prefer_mentioned_anchors,
        cfg.get("query_weighted_ppr", False), cfg.get("ppr_seed_temperature", 0.20),
        graph_edge_types,
        cfg.get("conditional_link_top_l", 2),
        cfg.get("conditional_link_min_score"),
        cfg.get("conditional_link_min_gain"),
        direct_all_sentence_pool,
    )
    candidate_cache = cfg.get("_candidate_cache")
    cached = candidate_cache.get(cache_key) if candidate_cache is not None else None
    if cached is None:
        candidate_start = time.perf_counter()
        if direct_all_sentence_pool:
            sentences = item.all_sentences()
            if cfg.get("use_reranker", False):
                scores = rerank_scores(
                    item.question, [sentence.embedding_text for sentence in sentences]
                )
                candidates = [
                    Candidate(sentence=sentence, semantic_score=score)
                    for sentence, score in zip(sentences, scores)
                ]
            else:
                q_vec = embed(item.question)
                vectors = embed_batch(
                    [sentence.embedding_text for sentence in sentences]
                )
                candidates = [
                    Candidate(
                        sentence=sentence,
                        semantic_score=cosine_sim(q_vec, vector),
                        embedding=vector,
                    )
                    for sentence, vector in zip(sentences, vectors)
                ]
        else:
            candidates = generate_candidates(
                item,
                max_hops=cfg.get("max_hops", 2),
                use_reranker=cfg.get("use_reranker", False),
                graph_rel_method=cfg.get("graph_rel_method", "hop"),
                conditional_rerank=conditional_rerank,
                conditional_anchor_n=conditional_anchor_n,
                prefer_mentioned_anchors=prefer_mentioned_anchors,
                query_weighted_ppr=cfg.get("query_weighted_ppr", False),
                ppr_seed_temperature=cfg.get("ppr_seed_temperature", 0.20),
                graph_edge_types=set(graph_edge_types) if graph_edge_types is not None else None,
                conditional_link_top_l=cfg.get("conditional_link_top_l", 2),
                conditional_link_min_score=cfg.get("conditional_link_min_score"),
                conditional_link_min_gain=cfg.get("conditional_link_min_gain"),
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
                    "preferred_titles": question_mentioned_titles(
                        item.question,
                        {candidate.sentence.passage_title for candidate in candidates},
                    ) if graft_v2 and route == DIRECT_COMPARISON else (),
                    "token_cost_power": cfg.get("token_cost_power", 1.0),
                    "min_marginal_score": cfg.get("min_marginal_score", 0.0),
                    "marginal_coverage_bonus": cfg.get("marginal_coverage_bonus", 0.0),
                    "witness_min_titles": cfg.get("witness_min_titles", 2),
                    "witness_chain_target": cfg.get(
                        "bridge_witness_chain_target", 2
                    ) if route == BRIDGE_COMPARISON else cfg.get(
                        "witness_chain_target", 1
                    ),
                    "witness_fallback_min_nodes": cfg.get(
                        "bridge_witness_fallback_min_nodes", 4
                    ) if route == BRIDGE_COMPARISON else cfg.get(
                        "witness_fallback_min_nodes", 2
                    ),
                    "witness_rescue_max_units": cfg.get(
                        "witness_rescue_max_units", 0
                    ),
                    "serialized_cost_aware": cfg.get(
                        "serialized_cost_aware", graft_v2
                    ),
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
    "wise_v3": _run_wise_v3,
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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _filter_qid_manifest(
    items: list[QAItem],
    manifest_path: str | Path,
    dataset_sha256: str,
) -> list[QAItem]:
    payload = json.loads(Path(manifest_path).read_text())
    qids = payload["qids"] if isinstance(payload, dict) else payload
    if isinstance(payload, dict) and payload.get("dataset_sha256") != dataset_sha256:
        raise ValueError(f"dataset checksum does not match manifest {manifest_path}")
    if len(qids) != len(set(qids)):
        raise ValueError(f"duplicate QIDs in manifest {manifest_path}")
    by_qid = {item.qid: item for item in items}
    missing = [qid for qid in qids if qid not in by_qid]
    if missing:
        raise ValueError(f"manifest {manifest_path} contains {len(missing)} unknown QIDs")
    return [by_qid[qid] for qid in qids]


def preflight(config_path: str) -> dict:
    """Validate config, dataset hygiene and manifests without loading models."""
    config = yaml.safe_load(Path(config_path).read_text())
    dataset_path = config.get("dataset", "data/raw/hotpot_dev_distractor_v1.json")
    dataset_sha256 = _sha256_file(dataset_path)
    items = load_hotpotqa(
        dataset_path,
        deduplicate_passages=config.get("deduplicate_passages", False),
        invalid_gold_policy=config.get("invalid_gold_policy", "keep"),
    )
    if config.get("qid_manifest"):
        items = _filter_qid_manifest(items, config["qid_manifest"], dataset_sha256)
    n_questions = config["n_questions"]
    if n_questions > len(items):
        raise ValueError(f"requested {n_questions} questions but only {len(items)} are available")

    names = [method["name"] for method in config["methods"]]
    if len(names) != len(set(names)):
        raise ValueError("method names must be unique within one config")
    for method in config["methods"]:
        if method["kind"] not in _RUNNERS:
            raise ValueError(f"unknown method kind: {method['kind']}")
        if method["kind"] == "wise_v3":
            default_backend = config.get("backend")
            _validate_wise_v3_config(
                method,
                default_planner_backend=config.get(
                    "llm_backend", default_backend or DEFAULT_LLM_BACKEND
                ),
                default_planner_model=config.get("llm_model", DEFAULT_MODEL),
            )
            continue
        strategy = method.get("strategy", "topk")
        if strategy not in set(VALID_STRATEGIES) | {
            "graft",
            "graft_v2",
            "routed_topk",
            "routed_graph_rescue",
        }:
            raise ValueError(f"unknown runner strategy: {strategy}")
        edge_types = set(method.get("graph_edge_types", ALL_EDGE_TYPES))
        if edge_types - ALL_EDGE_TYPES:
            raise ValueError(f"unknown edge types in method {method['name']}")
        if strategy == "graft_v2":
            for parameter in (
                "token_cost_power",
                "min_marginal_score",
                "marginal_coverage_bonus",
                "witness_rescue_max_units",
            ):
                if float(method.get(parameter, 0.0)) < 0:
                    raise ValueError(
                        f"{parameter} must be non-negative in method {method['name']}"
                    )
            for parameter, default in (
                ("witness_min_titles", 2),
                ("witness_chain_target", 1),
                ("bridge_witness_chain_target", 2),
                ("witness_fallback_min_nodes", 2),
                ("bridge_witness_fallback_min_nodes", 4),
            ):
                if int(method.get(parameter, default)) <= 0:
                    raise ValueError(
                        f"{parameter} must be positive in method {method['name']}"
                    )

    default_backend = config.get("backend")
    embedding_backend = config.get("embedding_backend", default_backend or "local")
    reranker_backend = config.get("reranker_backend", default_backend or "local")
    llm_backend = config.get(
        "llm_backend", default_backend or DEFAULT_LLM_BACKEND
    )
    if embedding_backend not in {"local", "openrouter"}:
        raise ValueError("embedding_backend must be 'local' or 'openrouter'")
    if reranker_backend not in {"local", "openrouter"}:
        raise ValueError("reranker_backend must be 'local' or 'openrouter'")
    if llm_backend not in {"openai", "openrouter"}:
        raise ValueError("llm_backend must be 'openai' or 'openrouter'")
    rows_per_question = sum(len(_expected_keys("preflight", method)) for method in config["methods"])
    report = {
        "config": config_path,
        "dataset": dataset_path,
        "dataset_sha256": dataset_sha256,
        "available_qids": len(items),
        "selected_qids": n_questions,
        "methods": len(config["methods"]),
        "rows_per_question": rows_per_question,
        "expected_rows": n_questions * rows_per_question,
        "expected_llm_calls": n_questions * rows_per_question if config.get("call_llm", False) else 0,
        "expected_planner_calls": n_questions
        * sum(method["kind"] == "wise_v3" for method in config["methods"]),
        "claim_verifier_call_policy": "one call per planned claim",
        "expected_dependency_verifier_calls_upper_bound": n_questions
        * sum(method["kind"] == "wise_v3" for method in config["methods"]),
        "embedding_backend": embedding_backend,
        "embedding_model": config.get("embedding_model"),
        "reranker_backend": reranker_backend,
        "reranker_model": config.get("reranker_model"),
        "llm_backend": llm_backend,
        "llm_model": config.get("llm_model", DEFAULT_MODEL),
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
    }
    print(json.dumps(report, indent=2))
    return report


def run(config_path: str) -> Path:
    config_text = Path(config_path).read_text()
    config = yaml.safe_load(config_text)
    config_sha256 = hashlib.sha256(config_text.encode()).hexdigest()
    dataset_path = config.get("dataset", "data/raw/hotpot_dev_distractor_v1.json")
    sample_seed = config.get("seed", 0)
    dataset_sha256 = _sha256_file(dataset_path)
    items = load_hotpotqa(
        dataset_path,
        deduplicate_passages=config.get("deduplicate_passages", False),
        invalid_gold_policy=config.get("invalid_gold_policy", "keep"),
    )
    qid_manifest = config.get("qid_manifest")
    if qid_manifest:
        items = _filter_qid_manifest(items, qid_manifest, dataset_sha256)
    if config["n_questions"] > len(items):
        raise ValueError(
            f"requested {config['n_questions']} questions but only {len(items)} are available"
        )
    items = _sample_items(
        items,
        config["n_questions"],
        sample_seed,
        stratified=config.get("stratified_sampling", False),
    )
    shard_count = int(os.environ.get("RIVF_SHARD_COUNT", "1"))
    shard_index = int(os.environ.get("RIVF_SHARD_INDEX", "0"))
    if shard_count <= 0:
        raise ValueError("RIVF_SHARD_COUNT must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("RIVF_SHARD_INDEX must be in [0, RIVF_SHARD_COUNT)")
    sharded_items = [
        item for position, item in enumerate(items) if position % shard_count == shard_index
    ]
    configured_run_limit = os.environ.get("RIVF_RUN_LIMIT")
    run_items = (
        sharded_items[: int(configured_run_limit)]
        if configured_run_limit
        else sharded_items
    )
    if configured_run_limit and int(configured_run_limit) <= 0:
        raise ValueError("RIVF_RUN_LIMIT must be positive")

    default_backend = config.get("backend")
    embedding_backend = config.get("embedding_backend", default_backend or "local")
    reranker_backend = config.get("reranker_backend", default_backend or "local")
    llm_backend = config.get(
        "llm_backend", default_backend or DEFAULT_LLM_BACKEND
    )
    configure_embeddings(embedding_backend, config.get("embedding_model"))
    configure_reranker(reranker_backend, config.get("reranker_model"))

    call_llm = config.get("call_llm", False) and os.environ.get("RIVF_SKIP_LLM") != "1"
    llm_model = config.get("llm_model", DEFAULT_MODEL)
    code_revision = _git_revision()
    git_dirty = _git_dirty()
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
        for i, item in enumerate(run_items, 1):
            pending: list[tuple[tuple, dict, str | None]] = []
            candidate_cache: dict[tuple, tuple[list[Candidate], float]] = {}
            for raw_method_cfg in config["methods"]:
                method_cfg = {
                    **raw_method_cfg,
                    "_candidate_cache": candidate_cache,
                    "_default_planner_backend": llm_backend,
                    "_default_planner_model": llm_model,
                }
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
                        "qid_manifest": qid_manifest,
                        "dataset_sha256": dataset_sha256,
                        "config_sha256": config_sha256,
                        "code_revision": code_revision,
                        "git_dirty": git_dirty,
                        "embedding_backend": embedding_backend_name(),
                        "embedding_model": embedding_model_name(),
                        "reranker_backend": (
                            reranker_backend_name()
                            if method_cfg.get("use_reranker", False)
                            else None
                        ),
                        "reranker_model": (
                            reranker_model_name()
                            if method_cfg.get("use_reranker", False)
                            else None
                        ),
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
                    chain_aware_routes = method_cfg.get("chain_aware_routes")
                    chain_aware_prompt = method_cfg.get("chain_aware_prompt", False)
                    if chain_aware_routes is not None:
                        router = (
                            route_question_v2
                            if method_cfg.get("strategy") == "graft_v2"
                            else route_question
                        )
                        chain_aware_prompt = router(item.question) in chain_aware_routes
                    reasoning_contract = None
                    if method_cfg["kind"] == "wise_v3" and not run_params.get(
                        "wise_failed"
                    ):
                        reasoning_contract = {
                            "plan": run_params.get("wise_plan"),
                            "claim_assignments": run_params.get(
                                "wise_claim_assignments"
                            ),
                            "verified_dependencies": run_params.get(
                                "wise_verified_dependencies"
                            ),
                        }
                    constructed_prompt = build_prompt(
                        item,
                        selected,
                        chain_aware=chain_aware_prompt,
                        preserve_selection_order=method_cfg.get(
                            "selection_order_prompt", False
                        ),
                        compact_selection_order=(
                            method_cfg["kind"] == "wise_v3"
                            or method_cfg.get("compact_selection_order_prompt", False)
                        ),
                        reasoning_contract=reasoning_contract,
                    )
                    row["constructed_prompt_tokens"] = count_tokens(constructed_prompt)
                    if run_params.get("wise_failed"):
                        row.update(
                            {
                                "prediction": "",
                                "llm_skipped_due_to_wise_error": True,
                                "llm_latency_s": 0.0,
                                "prompt_tokens": 0,
                                "output_tokens": 0,
                                **answer_metrics("", item.answer),
                            }
                        )
                        prompt = None
                    else:
                        prompt = constructed_prompt if call_llm else None
                    pending.append((key, row, prompt))
            if call_llm and pending:
                configured_workers = int(
                    os.environ.get("RIVF_LLM_CONCURRENCY", config.get("llm_concurrency", 6))
                )
                deduplicate_prompts = config.get(
                    "deduplicate_identical_prompts", False
                )
                generation_entries = []
                seen_prompts = set()
                for entry in pending:
                    prompt = entry[2]
                    if prompt is None:
                        continue
                    if not deduplicate_prompts or prompt not in seen_prompts:
                        generation_entries.append(entry)
                        seen_prompts.add(prompt)
                generated_by_prompt = {}
                if generation_entries:
                    workers = min(configured_workers, len(generation_entries))
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        fields = list(
                            executor.map(
                                lambda entry: _generate_fields(
                                    entry[2], llm_model, item.answer, llm_backend
                                ),
                                generation_entries,
                            )
                        )
                    generated_by_prompt = {
                        entry[2]: generated
                        for entry, generated in zip(generation_entries, fields)
                    }
                used_prompts = set()
                for _, row, prompt in pending:
                    if prompt is None:
                        continue
                    generated = dict(generated_by_prompt[prompt])
                    reused = deduplicate_prompts and prompt in used_prompts
                    generated["generation_reused"] = reused
                    if reused:
                        generated["llm_latency_s"] = 0.0
                        generated["api_prompt_tokens"] = 0
                        generated["api_completion_tokens"] = 0
                        generated["api_total_tokens"] = 0
                        generated["api_cost_usd"] = 0.0
                    row.update(generated)
                    used_prompts.add(prompt)
            for key, row, _ in pending:
                f.write(json.dumps(row) + "\n")
                f.flush()
                done_keys.add(key)
            if i % 20 == 0 or i == len(run_items):
                print(
                    f"...{i}/{len(sharded_items)} shard questions done"
                    + (
                        " (run limit reached)"
                        if len(run_items) < len(sharded_items)
                        else ""
                    )
                )

                metadata = {
                    "config": config_path,
                    "config_sha256": config_sha256,
                    "code_revision": code_revision,
                    "git_dirty": git_dirty,
                    "completed_rows": len(done_keys),
                    "shard_count": shard_count,
                    "shard_index": shard_index,
                    "embedding_backend": embedding_backend_name(),
                    "embedding_model": embedding_model_name(),
                    "embedding_api_usage": embedding_usage(),
                    "reranker_backend": reranker_backend_name(),
                    "reranker_model": reranker_model_name(),
                    "reranker_api_usage": reranker_usage(),
                    "llm_backend": llm_backend,
                    "llm_model": llm_model,
                }
                (out_path.parent / "run_metadata.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )

    print(f"Wrote results -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        preflight(args.config) if args.preflight else run(args.config)
    except WiseV3Error as exc:
        print(json.dumps({"error": exc.to_dict()}, ensure_ascii=False))
        raise SystemExit(2) from exc
