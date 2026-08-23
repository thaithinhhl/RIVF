import math
import random

from rivf.data.schema import Sentence
from rivf.eval.efficiency import count_tokens
from rivf.methods.base import Candidate
from rivf.retrieval.embeddings import cosine_sim

VALID_STRATEGIES = (
    "topk",
    "normalized_topk",
    "adaptive_mass",
    "coverage_topk",
    "conditional_topk",
    "graph_rescue_topk",
    "chain_set_topk",
    "threshold",
    "random",
    "rrf",
    "mmr",
    "interleave",
)
RRF_K = 60  # standard constant from Cormack et al. 2009
MMR_LAMBDA = 0.7  # weight on relevance vs. the redundancy penalty


def combined_score(c: Candidate, alpha: float) -> float:
    semantic = c.semantic_score if c.semantic_score is not None else 0.0
    graph = c.graph_score if c.graph_score is not None else 0.0
    return alpha * semantic + (1 - alpha) * graph


def percentile_scores(values: list[float]) -> list[float]:
    """Per-query percentile ranks in [0, 1], using average ranks for ties."""
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2
        for position in range(start, end):
            ranks[order[position]] = average_rank / (len(values) - 1)
        start = end
    return ranks


def normalized_component_scores(candidates: list[Candidate]) -> tuple[list[float], list[float]]:
    """Semantic and graph percentile ranks aligned with ``candidates``."""
    semantic = percentile_scores([c.semantic_score or 0.0 for c in candidates])
    graph = percentile_scores([c.graph_score or 0.0 for c in candidates])
    return semantic, graph


def normalized_combined_scores(candidates: list[Candidate], alpha: float) -> list[float]:
    semantic, graph = normalized_component_scores(candidates)
    return [alpha * sem + (1 - alpha) * rel for sem, rel in zip(semantic, graph)]


def optional_percentile_scores(values: list[float | None]) -> list[float]:
    """Percentile ranks for present values; missing optional signals get 0."""
    present = [index for index, value in enumerate(values) if value is not None]
    output = [0.0] * len(values)
    ranked = percentile_scores([float(values[index]) for index in present])
    for index, score in zip(present, ranked):
        output[index] = score
    return output


def _adaptive_mass_select(
    candidates: list[Candidate],
    alpha: float,
    budget_tokens: int | None,
    top_k: int | None,
    mass_threshold: float,
    mass_temperature: float,
    min_nodes: int,
    path_closure: bool,
) -> list[Sentence]:
    """Select the smallest high-score set covering a target softmax mass.

    Scores are per-query percentile-normalized before fusion. A hard token
    ceiling remains a safety constraint, but unlike fixed-budget Top-K this
    policy can stop early on confident questions. Optional path closure adds
    the intermediate nodes on one shortest seed-to-candidate path.
    """
    if not candidates:
        return []
    if not 0 < mass_threshold <= 1:
        raise ValueError("mass_threshold must be in (0, 1]")
    if mass_temperature <= 0:
        raise ValueError("mass_temperature must be positive")

    scores = normalized_combined_scores(candidates, alpha)
    peak = max(scores)
    weights = [math.exp((score - peak) / mass_temperature) for score in scores]
    total_weight = sum(weights)
    order = sorted(
        range(len(candidates)),
        key=lambda i: (scores[i], candidates[i].semantic_score or 0.0),
        reverse=True,
    )
    by_key = {candidate.sentence.key: candidate for candidate in candidates}
    selected: list[Candidate] = []
    selected_keys: set[tuple[str, int]] = set()
    used_tokens = 0
    covered_weight = 0.0

    for index in order:
        candidate = candidates[index]
        closure_keys = candidate.path_keys if path_closure and candidate.path_keys else (candidate.sentence.key,)
        additions = [by_key[key] for key in closure_keys if key in by_key and key not in selected_keys]
        added_cost = sum(count_tokens(c.sentence.text) for c in additions)
        if budget_tokens is not None and used_tokens + added_cost > budget_tokens:
            continue
        for addition in additions:
            selected.append(addition)
            selected_keys.add(addition.sentence.key)
        used_tokens += added_cost
        covered_weight += weights[index]
        enough_mass = covered_weight / total_weight >= mass_threshold
        if len(selected) >= min_nodes and enough_mass:
            break
        if top_k is not None and len(selected) >= top_k:
            break

    return [candidate.sentence for candidate in selected]


def _coverage_topk_select(
    candidates: list[Candidate],
    alpha: float,
    budget_tokens: int | None,
    top_k: int | None,
    coverage_bonus: float,
) -> list[Sentence]:
    """Greedily fill the budget while encouraging passage coverage.

    Multi-hop evidence normally spans multiple passages.  A small bonus for
    an as-yet unseen passage prevents several high-scoring sentences from one
    passage crowding out the other hop.  Relevance is percentile-normalized
    per query so the bonus has a stable scale.  Candidates that do not fit
    are skipped instead of ending selection and wasting the remaining budget.
    """
    if coverage_bonus < 0:
        raise ValueError("coverage_bonus must be non-negative")
    relevance = normalized_combined_scores(candidates, alpha)
    remaining = set(range(len(candidates)))
    selected: list[Sentence] = []
    covered_titles: set[str] = set()
    used_tokens = 0

    while remaining:
        fitting = [
            index
            for index in remaining
            if budget_tokens is None
            or used_tokens + count_tokens(candidates[index].sentence.text) <= budget_tokens
        ]
        if not fitting:
            break
        best = max(
            fitting,
            key=lambda index: (
                relevance[index]
                + (
                    coverage_bonus
                    if candidates[index].sentence.passage_title not in covered_titles
                    else 0.0
                ),
                relevance[index],
                candidates[index].semantic_score or 0.0,
            ),
        )
        candidate = candidates[best]
        selected.append(candidate.sentence)
        covered_titles.add(candidate.sentence.passage_title)
        used_tokens += count_tokens(candidate.sentence.text)
        remaining.remove(best)
        if top_k is not None and len(selected) >= top_k:
            break
    return selected


def _conditional_topk_select(
    candidates: list[Candidate],
    alpha: float,
    budget_tokens: int | None,
    top_k: int | None,
    conditional_weight: float,
) -> list[Sentence]:
    """Budget-fill using semantic relevance plus conditional second-hop CE.

    Selecting a rescued node also selects the anchor that produced its best
    conditional score, preserving the evidence pair rather than admitting an
    isolated answer-looking sentence.
    """
    if conditional_weight < 0:
        raise ValueError("conditional_weight must be non-negative")
    relevance = normalized_combined_scores(candidates, alpha)
    conditional = optional_percentile_scores(
        [candidate.conditional_score for candidate in candidates]
    )
    scores = [
        base + conditional_weight * rescue
        for base, rescue in zip(relevance, conditional)
    ]
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            scores[index], relevance[index], candidates[index].semantic_score or 0.0
        ),
        reverse=True,
    )
    by_key = {candidate.sentence.key: candidate for candidate in candidates}
    selected: list[Candidate] = []
    selected_keys: set[tuple[str, int]] = set()
    used_tokens = 0

    for index in order:
        candidate = candidates[index]
        pair_keys = (
            (candidate.conditional_anchor_key, candidate.sentence.key)
            if candidate.conditional_anchor_key is not None
            else (candidate.sentence.key,)
        )
        additions = [
            by_key[key]
            for key in pair_keys
            if key in by_key and key not in selected_keys
        ]
        added_cost = sum(count_tokens(addition.sentence.text) for addition in additions)
        if budget_tokens is not None and used_tokens + added_cost > budget_tokens:
            continue
        for addition in additions:
            selected.append(addition)
            selected_keys.add(addition.sentence.key)
        used_tokens += added_cost
        if top_k is not None and len(selected) >= top_k:
            break
    return [candidate.sentence for candidate in selected]


def graph_rescue_signals(candidates: list[Candidate], proxy_k: int = 6) -> dict[str, float]:
    """Inference-only diagnostics for graph-gated conditional rescue."""
    if not candidates:
        return {"disagreement": 0.0, "effective_weight_factor": 0.0}
    k = min(max(1, proxy_k), len(candidates))
    semantic_order = sorted(
        range(len(candidates)),
        key=lambda index: candidates[index].semantic_score or 0.0,
        reverse=True,
    )
    graph_order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index].graph_score or 0.0,
            candidates[index].semantic_score or 0.0,
        ),
        reverse=True,
    )
    semantic_keys = {candidates[index].sentence.key for index in semantic_order[:k]}
    graph_keys = {candidates[index].sentence.key for index in graph_order[:k]}
    union = semantic_keys | graph_keys
    overlap = len(semantic_keys & graph_keys) / len(union) if union else 1.0
    disagreement = 1.0 - overlap
    return {"disagreement": disagreement, "effective_weight_factor": disagreement}


def _graph_rescue_topk_select(
    candidates: list[Candidate],
    alpha: float,
    budget_tokens: int | None,
    top_k: int | None,
    conditional_weight: float,
    graph_rescue_weight: float,
    graph_rescue_k: int,
    graph_rescue_gate: bool,
    preserve_pairs: bool,
) -> list[Sentence]:
    """Use graph as a gated residual only for semantic Top-K misses.

    A candidate receives the extra graph term only when it is in graph Top-K,
    absent from semantic Top-K, and independently validated by conditional
    CE(q + anchor, v). Selecting it also retains its anchor.
    """
    if graph_rescue_weight < 0:
        raise ValueError("graph_rescue_weight must be non-negative")
    if graph_rescue_k <= 0:
        raise ValueError("graph_rescue_k must be positive")

    relevance = normalized_combined_scores(candidates, alpha)
    graph = percentile_scores([candidate.graph_score or 0.0 for candidate in candidates])
    conditional = optional_percentile_scores(
        [candidate.conditional_score for candidate in candidates]
    )
    k = min(graph_rescue_k, len(candidates))
    semantic_top = set(
        sorted(
            range(len(candidates)),
            key=lambda index: candidates[index].semantic_score or 0.0,
            reverse=True,
        )[:k]
    )
    graph_top = set(
        sorted(
            range(len(candidates)),
            key=lambda index: (
                candidates[index].graph_score or 0.0,
                candidates[index].semantic_score or 0.0,
            ),
            reverse=True,
        )[:k]
    )
    signals = graph_rescue_signals(candidates, proxy_k=k)
    effective_weight = graph_rescue_weight * (
        signals["effective_weight_factor"] if graph_rescue_gate else 1.0
    )
    scores = []
    for index, candidate in enumerate(candidates):
        rescue = 0.0
        if (
            index in graph_top
            and index not in semantic_top
            and candidate.conditional_score is not None
            and candidate.conditional_anchor_key is not None
        ):
            hop = candidate.hop_distance if candidate.hop_distance is not None else 1
            hop_penalty = 1.0 / max(1, hop)
            rescue = graph[index] * conditional[index] * hop_penalty
        scores.append(
            relevance[index]
            + conditional_weight * conditional[index]
            + effective_weight * rescue
        )

    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            scores[index], relevance[index], candidates[index].semantic_score or 0.0
        ),
        reverse=True,
    )
    by_key = {candidate.sentence.key: candidate for candidate in candidates}
    selected: list[Candidate] = []
    selected_keys: set[tuple[str, int]] = set()
    used_tokens = 0
    for index in order:
        candidate = candidates[index]
        keys = (candidate.sentence.key,)
        if preserve_pairs and candidate.conditional_anchor_key is not None:
            keys = (candidate.conditional_anchor_key, candidate.sentence.key)
        additions = [
            by_key[key] for key in keys if key in by_key and key not in selected_keys
        ]
        cost = sum(count_tokens(addition.sentence.text) for addition in additions)
        if budget_tokens is not None and used_tokens + cost > budget_tokens:
            continue
        for addition in additions:
            selected.append(addition)
            selected_keys.add(addition.sentence.key)
        used_tokens += cost
        if top_k is not None and len(selected) >= top_k:
            break
    return [candidate.sentence for candidate in selected]


def _chain_set_topk_select(
    candidates: list[Candidate],
    alpha: float,
    budget_tokens: int | None,
    top_k: int | None,
    conditional_weight: float,
) -> list[Sentence]:
    """Reserve one complete conditional pair per discovered anchor.

    Bridge-comparison questions contain two parallel chains (e.g. two films
    and their directors). Ranking sentences independently often spends the
    budget on one side. This selector first packs the strongest rescued
    endpoint for each anchor, then fills remaining space by the ordinary
    conditional score. No gold facts or benchmark type labels are used.
    """
    if conditional_weight < 0:
        raise ValueError("conditional_weight must be non-negative")
    relevance = normalized_combined_scores(candidates, alpha)
    conditional = optional_percentile_scores(
        [candidate.conditional_score for candidate in candidates]
    )
    scores = [
        base + conditional_weight * rescue
        for base, rescue in zip(relevance, conditional)
    ]
    by_key = {candidate.sentence.key: candidate for candidate in candidates}
    by_anchor: dict[tuple[str, int], list[int]] = {}
    for index, candidate in enumerate(candidates):
        if candidate.conditional_anchor_key is not None:
            by_anchor.setdefault(candidate.conditional_anchor_key, []).append(index)

    selected: list[Candidate] = []
    selected_keys: set[tuple[str, int]] = set()
    used_tokens = 0

    def try_add(keys: tuple[tuple[str, int], ...]) -> bool:
        nonlocal used_tokens
        additions = [
            by_key[key] for key in keys if key in by_key and key not in selected_keys
        ]
        cost = sum(count_tokens(candidate.sentence.text) for candidate in additions)
        if budget_tokens is not None and used_tokens + cost > budget_tokens:
            return False
        for addition in additions:
            selected.append(addition)
            selected_keys.add(addition.sentence.key)
        used_tokens += cost
        return True

    # Anchors themselves are selected by original semantic rank. Preserve
    # that order so the strongest chain gets first access to a tight budget.
    anchor_order = sorted(
        by_anchor,
        key=lambda key: by_key[key].semantic_score or 0.0,
        reverse=True,
    )
    for anchor_key in anchor_order:
        endpoint = max(
            by_anchor[anchor_key],
            key=lambda index: (scores[index], relevance[index]),
        )
        try_add((anchor_key, candidates[endpoint].sentence.key))
        if top_k is not None and len(selected) >= top_k:
            return [candidate.sentence for candidate in selected]

    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            scores[index], relevance[index], candidates[index].semantic_score or 0.0
        ),
        reverse=True,
    )
    for index in order:
        candidate = candidates[index]
        keys = (
            (candidate.conditional_anchor_key, candidate.sentence.key)
            if candidate.conditional_anchor_key is not None
            else (candidate.sentence.key,)
        )
        try_add(keys)
        if top_k is not None and len(selected) >= top_k:
            break
    return [candidate.sentence for candidate in selected]


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
    mass_threshold: float = 0.65,
    mass_temperature: float = 0.15,
    min_nodes: int = 4,
    path_closure: bool = False,
    coverage_bonus: float = 0.0,
    conditional_weight: float = 0.2,
    graph_rescue_weight: float = 0.0,
    graph_rescue_k: int = 6,
    graph_rescue_gate: bool = True,
    preserve_pairs: bool = True,
) -> list[Sentence]:
    """Select a subset of candidates.

    - "topk": rank by Score(v) = alpha*semantic + (1-alpha)*graph, descending.
    - "normalized_topk": percentile-normalize both signals within the query
      before applying the same weighted blend.
    - "adaptive_mass": normalized fusion followed by confidence-mass stopping;
      may also retain the shortest seed path for selected expanded nodes.
    - "coverage_topk": normalized relevance with a dynamic bonus for unseen
      passage titles; skips over candidates that do not fit the token budget.
    - "conditional_topk": adds CE(q + anchor, v) for graph-linked second-hop
      rescue and retains the corresponding anchor/evidence pair.
    - "graph_rescue_topk": adds a disagreement-gated GraphRel residual only
      to conditional-validated graph Top-K candidates missed by semantic Top-K.
    - "chain_set_topk": reserves one complete pair per conditional anchor,
      targeting two-chain bridge-comparison questions.
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

    if strategy == "adaptive_mass":
        return _adaptive_mass_select(
            candidates,
            alpha,
            budget_tokens,
            top_k,
            mass_threshold,
            mass_temperature,
            min_nodes,
            path_closure,
        )

    if strategy == "coverage_topk":
        return _coverage_topk_select(
            candidates, alpha, budget_tokens, top_k, coverage_bonus
        )

    if strategy == "conditional_topk":
        return _conditional_topk_select(
            candidates,
            alpha,
            budget_tokens,
            top_k,
            conditional_weight,
        )

    if strategy == "graph_rescue_topk":
        return _graph_rescue_topk_select(
            candidates,
            alpha,
            budget_tokens,
            top_k,
            conditional_weight,
            graph_rescue_weight,
            graph_rescue_k,
            graph_rescue_gate,
            preserve_pairs,
        )

    if strategy == "chain_set_topk":
        return _chain_set_topk_select(
            candidates,
            alpha,
            budget_tokens,
            top_k,
            conditional_weight,
        )

    if strategy == "random":
        ordered = list(candidates)
        random.Random(rng_seed).shuffle(ordered)
    elif strategy == "rrf":
        rrf = _rrf_scores(candidates, alpha)
        ordered = [c for _, c in sorted(zip(rrf, candidates), key=lambda pair: pair[0], reverse=True)]
    elif strategy == "normalized_topk":
        normalized = normalized_combined_scores(candidates, alpha)
        ordered = [
            c for _, c in sorted(zip(normalized, candidates), key=lambda pair: pair[0], reverse=True)
        ]
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
