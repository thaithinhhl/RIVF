import math
import random

from rivf.data.schema import Sentence
from rivf.eval.efficiency import count_tokens
from rivf.generation.prompts import build_context_compact_selection_order
from rivf.methods.base import Candidate
from rivf.retrieval.embeddings import cosine_sim

VALID_STRATEGIES = (
    "topk",
    "normalized_topk",
    "adaptive_mass",
    "coverage_topk",
    "conditional_topk",
    "graft_pair_topk",
    "graft_v2_pair",
    "graft_v2_witness",
    "entity_balanced_topk",
    "entity_witness",
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


def _graft_pair_topk_select(
    candidates: list[Candidate],
    budget_tokens: int | None,
    top_k: int | None,
    conditional_weight: float,
    preserve_pairs: bool,
) -> list[Sentence]:
    """Select GRAFT evidence units under a hard token ceiling.

    A validated endpoint is represented by the atomic unit ``{anchor, v}``
    with utility

        node(anchor) + node(v) + beta * link(anchor, v).

    All components are percentile-normalized within the question. Candidates
    without a validated link remain singleton units scored by node relevance.
    If an atomic pair does not fit, the entire pair is skipped. Setting
    ``preserve_pairs=False`` retains the same pair utility but admits only the
    endpoint, providing a controlled chain-preservation ablation.
    """
    if conditional_weight < 0:
        raise ValueError("conditional_weight must be non-negative")
    if not candidates:
        return []

    node_scores = percentile_scores(
        [candidate.semantic_score or 0.0 for candidate in candidates]
    )
    link_scores = optional_percentile_scores(
        [
            candidate.conditional_score
            if candidate.conditional_validated
            else None
            for candidate in candidates
        ]
    )
    by_key = {candidate.sentence.key: candidate for candidate in candidates}
    index_by_key = {
        candidate.sentence.key: index for index, candidate in enumerate(candidates)
    }
    units: list[tuple[float, float, tuple[tuple[str, int], ...]]] = []

    for index, candidate in enumerate(candidates):
        anchor_key = candidate.conditional_anchor_key
        if (
            candidate.conditional_validated
            and anchor_key is not None
            and anchor_key in index_by_key
        ):
            anchor_index = index_by_key[anchor_key]
            utility = (
                node_scores[anchor_index]
                + node_scores[index]
                + conditional_weight * link_scores[index]
            )
            keys = (
                (anchor_key, candidate.sentence.key)
                if preserve_pairs
                else (candidate.sentence.key,)
            )
        else:
            utility = node_scores[index]
            keys = (candidate.sentence.key,)
        units.append((utility, node_scores[index], keys))

    units.sort(key=lambda unit: (unit[0], unit[1]), reverse=True)
    selected: list[Sentence] = []
    selected_keys: set[tuple[str, int]] = set()
    used_tokens = 0

    for _, _, keys in units:
        additions = [by_key[key] for key in keys if key not in selected_keys]
        if not additions:
            continue
        if top_k is not None and len(selected) + len(additions) > top_k:
            continue
        cost = sum(count_tokens(candidate.sentence.text) for candidate in additions)
        if budget_tokens is not None and used_tokens + cost > budget_tokens:
            continue
        for addition in additions:
            selected.append(addition.sentence)
            selected_keys.add(addition.sentence.key)
        used_tokens += cost
        if top_k is not None and len(selected) >= top_k:
            break

    return selected


def _marginal_score(utility: float, token_cost: int, token_cost_power: float) -> float:
    if token_cost_power < 0:
        raise ValueError("token_cost_power must be non-negative")
    return utility / (max(1, token_cost) ** token_cost_power)


def _incremental_context_cost(
    selected: list[Sentence],
    additions: list[Candidate],
    serialized_cost_aware: bool,
) -> int:
    """Cost used for ranking; hard ceilings still count sentence text only."""
    if not serialized_cost_aware:
        return sum(count_tokens(addition.sentence.text) for addition in additions)
    before = count_tokens(build_context_compact_selection_order(selected))
    after = count_tokens(
        build_context_compact_selection_order(
            [*selected, *(addition.sentence for addition in additions)]
        )
    )
    return max(1, after - before)


def _graft_v2_pair_select(
    candidates: list[Candidate],
    budget_tokens: int | None,
    top_k: int | None,
    conditional_weight: float,
    token_cost_power: float,
    min_marginal_score: float,
    marginal_coverage_bonus: float,
) -> list[Sentence]:
    """Token-aware atomic chain selection for GRAFT-v2.

    Unlike v1's static unit ordering, utility is recomputed against the
    already-selected set. Reused anchors therefore have zero additional node
    utility and zero token cost, preventing repeated anchor credit. Units are
    ranked by marginal utility per incremental token and selection can stop
    before the hard ceiling when no remaining unit clears the configured
    score threshold.
    """
    if conditional_weight < 0:
        raise ValueError("conditional_weight must be non-negative")
    if min_marginal_score < 0:
        raise ValueError("min_marginal_score must be non-negative")
    if marginal_coverage_bonus < 0:
        raise ValueError("marginal_coverage_bonus must be non-negative")
    if not candidates:
        return []

    node_scores = percentile_scores(
        [candidate.semantic_score or 0.0 for candidate in candidates]
    )
    link_values = [
        (
            candidate.conditional_gain
            if candidate.conditional_gain is not None
            else candidate.conditional_score
        )
        if candidate.conditional_validated
        else None
        for candidate in candidates
    ]
    link_scores = optional_percentile_scores(link_values)
    by_key = {candidate.sentence.key: candidate for candidate in candidates}
    index_by_key = {
        candidate.sentence.key: index for index, candidate in enumerate(candidates)
    }
    units: list[tuple[tuple[tuple[str, int], ...], float, float]] = []
    for index, candidate in enumerate(candidates):
        anchor_key = candidate.conditional_anchor_key
        if (
            candidate.conditional_validated
            and anchor_key is not None
            and anchor_key in index_by_key
        ):
            keys = (anchor_key, candidate.sentence.key)
            link_utility = conditional_weight * link_scores[index]
        else:
            keys = (candidate.sentence.key,)
            link_utility = 0.0
        units.append((keys, link_utility, node_scores[index]))

    selected: list[Sentence] = []
    selected_keys: set[tuple[str, int]] = set()
    covered_titles: set[str] = set()
    used_tokens = 0

    while True:
        feasible = []
        for keys, link_utility, endpoint_score in units:
            additions = [by_key[key] for key in keys if key not in selected_keys]
            if not additions:
                continue
            if top_k is not None and len(selected) + len(additions) > top_k:
                continue
            cost = sum(count_tokens(addition.sentence.text) for addition in additions)
            if budget_tokens is not None and used_tokens + cost > budget_tokens:
                continue
            new_titles = {
                addition.sentence.passage_title for addition in additions
            } - covered_titles
            node_utility = sum(
                node_scores[index_by_key[addition.sentence.key]]
                for addition in additions
            )
            utility = (
                node_utility
                + link_utility
                + marginal_coverage_bonus * len(new_titles)
            )
            score = _marginal_score(utility, cost, token_cost_power)
            feasible.append(
                (
                    score,
                    utility,
                    endpoint_score,
                    tuple(addition.sentence.key for addition in additions),
                    additions,
                    cost,
                )
            )
        if not feasible:
            break
        best = max(feasible, key=lambda entry: entry[:4])
        if best[0] < min_marginal_score:
            break
        additions, cost = best[4], best[5]
        for addition in additions:
            selected.append(addition.sentence)
            selected_keys.add(addition.sentence.key)
            covered_titles.add(addition.sentence.passage_title)
        used_tokens += cost
        if top_k is not None and len(selected) >= top_k:
            break

    return selected


def _entity_balanced_topk_select(
    candidates: list[Candidate],
    budget_tokens: int | None,
    top_k: int | None,
    preferred_titles: tuple[str, ...],
    token_cost_power: float,
    min_marginal_score: float,
) -> list[Sentence]:
    """Direct-comparison selector that reserves evidence for named entities."""
    if min_marginal_score < 0:
        raise ValueError("min_marginal_score must be non-negative")
    if not candidates:
        return []
    relevance = percentile_scores(
        [candidate.semantic_score or 0.0 for candidate in candidates]
    )
    indexed = list(enumerate(candidates))
    selected: list[Sentence] = []
    selected_keys: set[tuple[str, int]] = set()
    used_tokens = 0

    def try_add(index: int, *, reserve: bool = False) -> bool:
        nonlocal used_tokens
        candidate = candidates[index]
        if candidate.sentence.key in selected_keys:
            return False
        if top_k is not None and len(selected) >= top_k:
            return False
        cost = count_tokens(candidate.sentence.text)
        if budget_tokens is not None and used_tokens + cost > budget_tokens:
            return False
        score = _marginal_score(relevance[index], cost, token_cost_power)
        if not reserve and score < min_marginal_score:
            return False
        selected.append(candidate.sentence)
        selected_keys.add(candidate.sentence.key)
        used_tokens += cost
        return True

    # Reserve the best direct evidence sentence for each explicitly-mentioned
    # passage before filling the remaining budget by token-aware relevance.
    for title in preferred_titles:
        matches = [
            index
            for index, candidate in indexed
            if candidate.sentence.passage_title == title
        ]
        if matches:
            best = max(matches, key=lambda index: relevance[index])
            try_add(best, reserve=True)

    remaining = sorted(
        (
            (
                _marginal_score(
                    relevance[index],
                    count_tokens(candidate.sentence.text),
                    token_cost_power,
                ),
                relevance[index],
                candidate.sentence.key,
                index,
            )
            for index, candidate in indexed
            if candidate.sentence.key not in selected_keys
        ),
        reverse=True,
    )
    for score, _, _, index in remaining:
        if score < min_marginal_score:
            break
        try_add(index)
        if top_k is not None and len(selected) >= top_k:
            break
    return selected


def _entity_witness_select(
    candidates: list[Candidate],
    budget_tokens: int | None,
    top_k: int | None,
    preferred_titles: tuple[str, ...],
    token_cost_power: float,
    min_marginal_score: float,
    witness_min_titles: int,
    witness_rescue_max_units: int,
    serialized_cost_aware: bool,
) -> list[Sentence]:
    """Select the smallest direct-comparison witness, then stop.

    One best sentence is reserved for each grounded comparison entity. If
    exact title grounding is incomplete, high-relevance distinct passages
    fill the missing roles. Optional rescue units are tightly bounded instead
    of filling the remaining hard budget.
    """
    if witness_min_titles <= 0:
        raise ValueError("witness_min_titles must be positive")
    if witness_rescue_max_units < 0:
        raise ValueError("witness_rescue_max_units must be non-negative")
    if not candidates:
        return []

    relevance = percentile_scores(
        [candidate.semantic_score or 0.0 for candidate in candidates]
    )
    selected: list[Sentence] = []
    selected_keys: set[tuple[str, int]] = set()
    covered_titles: set[str] = set()
    used_tokens = 0

    def fitting(index: int) -> bool:
        candidate = candidates[index]
        raw_cost = count_tokens(candidate.sentence.text)
        return (
            candidate.sentence.key not in selected_keys
            and (top_k is None or len(selected) < top_k)
            and (budget_tokens is None or used_tokens + raw_cost <= budget_tokens)
        )

    def add(index: int) -> None:
        nonlocal used_tokens
        candidate = candidates[index]
        selected.append(candidate.sentence)
        selected_keys.add(candidate.sentence.key)
        covered_titles.add(candidate.sentence.passage_title)
        used_tokens += count_tokens(candidate.sentence.text)

    # A direct comparison normally needs two entity/attribute facts. Limit
    # grounding to that structural requirement even if extra titles happen to
    # occur verbatim in the question.
    for title in preferred_titles[:witness_min_titles]:
        matches = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.sentence.passage_title == title and fitting(index)
        ]
        if matches:
            add(max(matches, key=lambda index: relevance[index]))

    # Alias or punctuation mismatches can prevent exact grounding. Recover the
    # missing comparison roles from distinct high-relevance passages.
    while len(covered_titles) < witness_min_titles:
        choices = [
            index
            for index, candidate in enumerate(candidates)
            if fitting(index)
            and candidate.sentence.passage_title not in covered_titles
        ]
        if not choices:
            break
        best = max(
            choices,
            key=lambda index: (
                relevance[index],
                candidates[index].semantic_score or 0.0,
                candidates[index].sentence.key,
            ),
        )
        add(best)

    # Rescue is opt-in and bounded. Required witness facts above never depend
    # on an arbitrary score threshold; only optional additions do.
    for _ in range(witness_rescue_max_units):
        choices = []
        for index, candidate in enumerate(candidates):
            if not fitting(index):
                continue
            incremental_cost = _incremental_context_cost(
                selected, [candidate], serialized_cost_aware
            )
            score = _marginal_score(
                relevance[index], incremental_cost, token_cost_power
            )
            choices.append((score, relevance[index], candidate.sentence.key, index))
        if not choices:
            break
        best = max(choices)
        if best[0] < min_marginal_score:
            break
        add(best[3])
    return selected


def _graft_v2_witness_select(
    candidates: list[Candidate],
    budget_tokens: int | None,
    top_k: int | None,
    conditional_weight: float,
    preferred_titles: tuple[str, ...],
    token_cost_power: float,
    min_marginal_score: float,
    witness_chain_target: int,
    witness_fallback_min_nodes: int,
    witness_rescue_max_units: int,
    serialized_cost_aware: bool,
) -> list[Sentence]:
    """Select a minimal set of complete anchor-endpoint witnesses.

    Chained questions stop after one validated pair; bridge comparisons can
    request two pairs from distinct grounded anchors. Singleton relevance is
    used only as a fallback when graph verification cannot complete the
    requested witness structure.
    """
    if conditional_weight < 0:
        raise ValueError("conditional_weight must be non-negative")
    if witness_chain_target <= 0:
        raise ValueError("witness_chain_target must be positive")
    if witness_fallback_min_nodes <= 0:
        raise ValueError("witness_fallback_min_nodes must be positive")
    if witness_rescue_max_units < 0:
        raise ValueError("witness_rescue_max_units must be non-negative")
    if not candidates:
        return []

    node_scores = percentile_scores(
        [candidate.semantic_score or 0.0 for candidate in candidates]
    )
    link_values = [
        candidate.conditional_gain
        if candidate.conditional_validated
        else None
        for candidate in candidates
    ]
    link_scores = optional_percentile_scores(link_values)
    by_key = {candidate.sentence.key: candidate for candidate in candidates}
    index_by_key = {
        candidate.sentence.key: index for index, candidate in enumerate(candidates)
    }
    pair_units = []
    for endpoint_index, endpoint in enumerate(candidates):
        anchor_key = endpoint.conditional_anchor_key
        if (
            not endpoint.conditional_validated
            or anchor_key is None
            or anchor_key not in index_by_key
        ):
            continue
        anchor = by_key[anchor_key]
        pair_units.append(
            {
                "keys": (anchor_key, endpoint.sentence.key),
                "anchor_title": anchor.sentence.passage_title,
                "link_utility": conditional_weight * link_scores[endpoint_index],
                "endpoint_score": node_scores[endpoint_index],
            }
        )

    selected: list[Sentence] = []
    selected_keys: set[tuple[str, int]] = set()
    covered_anchor_titles: set[str] = set()
    covered_titles: set[str] = set()
    used_tokens = 0

    def additions_for(keys: tuple[tuple[str, int], ...]) -> list[Candidate]:
        return [by_key[key] for key in keys if key not in selected_keys]

    def feasible(additions: list[Candidate]) -> bool:
        raw_cost = sum(count_tokens(addition.sentence.text) for addition in additions)
        return bool(additions) and (
            top_k is None or len(selected) + len(additions) <= top_k
        ) and (
            budget_tokens is None or used_tokens + raw_cost <= budget_tokens
        )

    def add_candidates(additions: list[Candidate]) -> None:
        nonlocal used_tokens
        for addition in additions:
            selected.append(addition.sentence)
            selected_keys.add(addition.sentence.key)
            covered_titles.add(addition.sentence.passage_title)
            used_tokens += count_tokens(addition.sentence.text)

    def ranked_pairs(required_anchor_title: str | None = None):
        ranked = []
        for unit_index, unit in enumerate(pair_units):
            if unit["anchor_title"] in covered_anchor_titles:
                continue
            if (
                required_anchor_title is not None
                and unit["anchor_title"] != required_anchor_title
            ):
                continue
            additions = additions_for(unit["keys"])
            if not feasible(additions):
                continue
            node_utility = sum(
                node_scores[index_by_key[addition.sentence.key]]
                for addition in additions
            )
            utility = node_utility + unit["link_utility"]
            incremental_cost = _incremental_context_cost(
                selected, additions, serialized_cost_aware
            )
            score = _marginal_score(utility, incremental_cost, token_cost_power)
            ranked.append(
                (
                    score,
                    utility,
                    unit["endpoint_score"],
                    unit["keys"],
                    unit_index,
                    additions,
                )
            )
        return ranked

    # Grounded bridge comparisons require one branch per named source entity.
    for title in preferred_titles[:witness_chain_target]:
        choices = ranked_pairs(title)
        if not choices:
            continue
        best = max(choices, key=lambda entry: entry[:5])
        unit = pair_units[best[4]]
        add_candidates(best[5])
        covered_anchor_titles.add(unit["anchor_title"])

    while len(covered_anchor_titles) < witness_chain_target:
        choices = ranked_pairs()
        if not choices:
            break
        best = max(choices, key=lambda entry: entry[:5])
        unit = pair_units[best[4]]
        add_candidates(best[5])
        covered_anchor_titles.add(unit["anchor_title"])

    # If graph verification fails, retain only the minimum semantic evidence
    # needed to expose distinct passages, rather than falling back to fill.
    missing_verified_witness = (
        len(covered_anchor_titles) < witness_chain_target
    )
    while missing_verified_witness and len(selected) < witness_fallback_min_nodes:
        choices = []
        for index, candidate in enumerate(candidates):
            if candidate.sentence.key in selected_keys:
                continue
            additions = [candidate]
            if not feasible(additions):
                continue
            coverage = candidate.sentence.passage_title not in covered_titles
            incremental_cost = _incremental_context_cost(
                selected, additions, serialized_cost_aware
            )
            score = _marginal_score(
                node_scores[index], incremental_cost, token_cost_power
            )
            choices.append(
                (
                    coverage,
                    score,
                    node_scores[index],
                    candidate.sentence.key,
                    additions,
                )
            )
        if not choices:
            break
        add_candidates(max(choices, key=lambda entry: entry[:4])[4])

    # Optional rescue remains bounded and thresholded. Defaults choose no
    # rescue, making actual usage depend on witness size rather than Bmax.
    for _ in range(witness_rescue_max_units):
        choices = ranked_pairs()
        if not choices:
            break
        best = max(choices, key=lambda entry: entry[:5])
        if best[0] < min_marginal_score:
            break
        unit = pair_units[best[4]]
        add_candidates(best[5])
        covered_anchor_titles.add(unit["anchor_title"])

    return selected


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
    preferred_titles: tuple[str, ...] = (),
    token_cost_power: float = 1.0,
    min_marginal_score: float = 0.0,
    marginal_coverage_bonus: float = 0.0,
    witness_min_titles: int = 2,
    witness_chain_target: int = 1,
    witness_fallback_min_nodes: int = 2,
    witness_rescue_max_units: int = 0,
    serialized_cost_aware: bool = False,
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
    - "graft_pair_topk": ranks validated anchor-endpoint units by the GRAFT
      utility and keeps each pair atomically under the token budget.
    - "graft_v2_pair": recomputes atomic-unit marginal utility against the
      selected set, normalizes it by incremental token cost, and can stop
      before filling the hard budget.
    - "graft_v2_witness": selects only the complete chain witnesses required
      by the question route, with bounded optional rescue.
    - "entity_balanced_topk": reserves direct evidence for passage titles
      explicitly mentioned in the question, then fills by relevance/token.
    - "entity_witness": selects one fact per comparison entity and stops once
      the direct-comparison witness is complete.
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

    if strategy == "graft_pair_topk":
        return _graft_pair_topk_select(
            candidates,
            budget_tokens,
            top_k,
            conditional_weight,
            preserve_pairs,
        )

    if strategy == "graft_v2_pair":
        return _graft_v2_pair_select(
            candidates,
            budget_tokens,
            top_k,
            conditional_weight,
            token_cost_power,
            min_marginal_score,
            marginal_coverage_bonus,
        )

    if strategy == "graft_v2_witness":
        return _graft_v2_witness_select(
            candidates,
            budget_tokens,
            top_k,
            conditional_weight,
            preferred_titles,
            token_cost_power,
            min_marginal_score,
            witness_chain_target,
            witness_fallback_min_nodes,
            witness_rescue_max_units,
            serialized_cost_aware,
        )

    if strategy == "entity_balanced_topk":
        return _entity_balanced_topk_select(
            candidates,
            budget_tokens,
            top_k,
            preferred_titles,
            token_cost_power,
            min_marginal_score,
        )

    if strategy == "entity_witness":
        return _entity_witness_select(
            candidates,
            budget_tokens,
            top_k,
            preferred_titles,
            token_cost_power,
            min_marginal_score,
            witness_min_titles,
            witness_rescue_max_units,
            serialized_cost_aware,
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
