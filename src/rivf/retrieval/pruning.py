import random

from rivf.data.schema import Sentence
from rivf.eval.efficiency import count_tokens
from rivf.methods.base import Candidate
from rivf.retrieval.embeddings import cosine_sim

VALID_STRATEGIES = ("topk", "threshold", "random", "rrf", "mmr", "interleave")
RRF_K = 60  # standard constant from Cormack et al. 2009
MMR_LAMBDA = 0.7  # weight on relevance vs. the redundancy penalty


def combined_score(c: Candidate, alpha: float) -> float:
    semantic = c.semantic_score if c.semantic_score is not None else 0.0
    graph = c.graph_score if c.graph_score is not None else 0.0
    return alpha * semantic + (1 - alpha) * graph


def _rrf_scores(candidates: list[Candidate], alpha: float, k: int = RRF_K) -> list[float]:
    """Reciprocal Rank Fusion: rank by each signal separately, combine by
    rank rather than raw value. Sidesteps the scale mismatch between
    semantic_score (a narrow-band cosine similarity) and graph_score (a
    coarse, discrete value) that a direct weighted sum is exposed to.
    Returns one RRF score per candidate, aligned by index with the input."""
    n = len(candidates)
    order_semantic = sorted(range(n), key=lambda i: candidates[i].semantic_score or 0.0, reverse=True)
    order_graph = sorted(range(n), key=lambda i: candidates[i].graph_score or 0.0, reverse=True)
    rank_semantic = [0] * n
    rank_graph = [0] * n
    for r, idx in enumerate(order_semantic, start=1):
        rank_semantic[idx] = r
    for r, idx in enumerate(order_graph, start=1):
        rank_graph[idx] = r
    return [
        alpha / (k + rank_semantic[i]) + (1 - alpha) / (k + rank_graph[i]) for i in range(n)
    ]


def _mmr_select(
    candidates: list[Candidate],
    alpha: float,
    budget_tokens: int | None,
    top_k: int | None,
    lambda_param: float = MMR_LAMBDA,
) -> list[Sentence]:
    """Greedy Maximal Marginal Relevance: iteratively pick the candidate
    that is both relevant (combined_score) and diverse from what's already
    picked (low max-similarity to already-selected embeddings). Targets a
    specific failure mode of plain top-K at a tight budget: several
    near-duplicate sentences about the same sub-topic can crowd out the
    *other* hop of a multi-hop question. Same simple stop-at-first-overflow
    budget rule as the other strategies -- not a knapsack solve."""
    remaining = list(candidates)
    selected: list[Candidate] = []
    used_tokens = 0

    while remaining:
        if selected:
            def mmr_score(c: Candidate) -> float:
                relevance = combined_score(c, alpha)
                if c.embedding is None:
                    return relevance
                redundancy = max(
                    (cosine_sim(c.embedding, s.embedding) for s in selected if s.embedding is not None),
                    default=0.0,
                )
                return lambda_param * relevance - (1 - lambda_param) * redundancy

            best_idx = max(range(len(remaining)), key=lambda i: mmr_score(remaining[i]))
        else:
            best_idx = max(range(len(remaining)), key=lambda i: combined_score(remaining[i], alpha))

        candidate = remaining[best_idx]
        cost = count_tokens(candidate.sentence.text)
        if budget_tokens is not None and used_tokens + cost > budget_tokens:
            break
        selected.append(candidate)
        used_tokens += cost
        del remaining[best_idx]
        if top_k is not None and len(selected) >= top_k:
            break

    return [c.sentence for c in selected]


def _interleave_select(
    candidates: list[Candidate], budget_tokens: int | None, top_k: int | None
) -> list[Sentence]:
    """Alternately pick the next unselected candidate by semantic_score,
    then by graph_score, instead of ranking by one linear-combined score.

    Targets a specific failure mode found via a stratified diagnostic split
    (questions labeled semantic-sufficient / semantic-blind / unrecoverable
    by whether a semantic-only or semantic+graph-union selection covers the
    gold evidence -- see version1.md sec. 10): graph_score only takes 3
    discrete values (1/(1+hop) for hop=0,1,2), so its fixed contribution to
    the blended score can let a barely-relevant seed sentence (hop=0, so
    graph_score=1.0) outrank a moderately-relevant multi-hop sentence that
    a semantic-only ranking would have kept. Giving both signals their own
    turn, rather than blending them into one number, rescued the targeted
    "semantic-blind" bucket's recall substantially -- but at a real cost to
    overall Precision/F1 (more candidates admitted just because it was that
    channel's turn), so this is an ablation/diagnostic strategy, not a
    replacement for the "topk" default. alpha is unused -- the two channels
    get equal, fixed turns rather than a weighted blend.

    A pick that would overflow the budget is skipped (not a stopping
    condition) so a later, cheaper pick from either channel still gets a
    chance -- unlike "topk"'s stop-at-first-overflow cutoff."""
    sem_order = sorted(candidates, key=lambda c: c.semantic_score or 0.0, reverse=True)
    graph_order = sorted(
        candidates, key=lambda c: (c.graph_score or 0.0, c.semantic_score or 0.0), reverse=True
    )
    selected: list[Candidate] = []
    selected_keys: set = set()
    used_tokens = 0
    i = j = 0
    turn_semantic = True
    while i < len(sem_order) or j < len(graph_order):
        order, idx = (sem_order, i) if turn_semantic else (graph_order, j)
        candidate = None
        while idx < len(order):
            if order[idx].sentence.key not in selected_keys:
                candidate = order[idx]
                idx += 1
                break
            idx += 1
        if turn_semantic:
            i = idx
        else:
            j = idx
        turn_semantic = not turn_semantic
        if candidate is None:
            continue
        selected_keys.add(candidate.sentence.key)
        cost = count_tokens(candidate.sentence.text)
        if budget_tokens is not None and used_tokens + cost > budget_tokens:
            continue
        selected.append(candidate)
        used_tokens += cost
        if top_k is not None and len(selected) >= top_k:
            break
    return [c.sentence for c in selected]


def prune(
    candidates: list[Candidate],
    alpha: float = 1.0,
    strategy: str = "topk",
    budget_tokens: int | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
    rng_seed: int = 0,
) -> list[Sentence]:
    """Select a subset of candidates.

    - "topk": rank by Score(v) = alpha*semantic + (1-alpha)*graph, descending.
    - "rrf": rank by Reciprocal Rank Fusion of the two signals' separate
      rankings, descending -- avoids the scale mismatch a direct weighted
      sum is exposed to (see _rrf_scores).
    - "threshold": keep every candidate with Score(v) >= `threshold` (ignores
      budget_tokens/top_k).
    - "random": shuffle, ignoring scores entirely -- the ablation sanity check.
    - "interleave": alternate turns between a semantic-only ranking and a
      graph-only ranking instead of blending them into one score (see
      _interleave_select) -- a diagnostic strategy, not the default.

    "topk"/"rrf"/"random" then cut by `budget_tokens` (greedy: add in rank
    order, stop at the first candidate that would exceed the budget -- a
    simple cutoff, not a knapsack solve) or by `top_k` count if no token
    budget is given. If neither is given, every candidate survives -- this
    is how a "full expansion, no pruning" baseline is expressed.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"unknown pruning strategy: {strategy!r}")

    if strategy == "threshold":
        if threshold is None:
            raise ValueError("strategy='threshold' requires a threshold value")
        return [c.sentence for c in candidates if combined_score(c, alpha) >= threshold]

    if strategy == "mmr":
        # Selection and the budget cutoff are interleaved (redundancy depends
        # on what's already picked), unlike the other strategies' sort-then-cut.
        return _mmr_select(candidates, alpha, budget_tokens, top_k)

    if strategy == "interleave":
        return _interleave_select(candidates, budget_tokens, top_k)

    if strategy == "random":
        ordered = list(candidates)
        random.Random(rng_seed).shuffle(ordered)
    elif strategy == "rrf":
        rrf = _rrf_scores(candidates, alpha)
        ordered = [c for _, c in sorted(zip(rrf, candidates), key=lambda pair: pair[0], reverse=True)]
    else:  # topk
        ordered = sorted(candidates, key=lambda c: combined_score(c, alpha), reverse=True)

    if budget_tokens is not None:
        selected: list[Sentence] = []
        used = 0
        for c in ordered:
            t = count_tokens(c.sentence.text)
            if used + t > budget_tokens:
                break
            selected.append(c.sentence)
            used += t
        return selected

    if top_k is not None:
        return [c.sentence for c in ordered[:top_k]]

    return [c.sentence for c in ordered]
