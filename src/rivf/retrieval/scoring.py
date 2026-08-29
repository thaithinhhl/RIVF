import math
import re

from rivf.data.schema import QAItem
from rivf.graph.build import (
    bfs_hop_distances_with_parents,
    build_graph,
    personalized_pagerank_scores,
)
from rivf.methods.base import Candidate
from rivf.retrieval.embeddings import cosine_sim, embed, embed_batch
from rivf.retrieval.reranker import rerank_scores
from rivf.retrieval.seed import DEFAULT_SEED_N, seed_retrieve


def generate_candidates(
    item: QAItem,
    max_hops: int = 2,
    seed_n: int = DEFAULT_SEED_N,
    use_reranker: bool = False,
    graph_rel_method: str = "hop",
    conditional_rerank: bool = False,
    conditional_anchor_n: int = 2,
    prefer_mentioned_anchors: bool = False,
    query_weighted_ppr: bool = False,
    ppr_seed_temperature: float = 0.20,
    graph_edge_types: set[str] | frozenset[str] | None = None,
    conditional_link_top_l: int | None = 2,
    conditional_link_min_score: float | None = None,
) -> list[Candidate]:
    """Seed retrieval -> graph expansion -> per-candidate semantic_score +
    graph_score, computed once.

    `graph_rel_method="hop"` (default): GraphRel(v) = 1/(1+hop_distance),
    the simplest possible seed-proximity signal, per the proposal's own
    start-from-the-smallest-version principle. `graph_rel_method="ppr"`:
    GraphRel(v) = Personalized PageRank mass from the seeds, renormalized
    to the candidate pool's max so it's on the same 0-1 footing as the
    hop-based score -- continuous and sensitive to multiple paths, unlike
    hop-distance's single shortest path (see graph/build.py). The
    candidate POOL itself (which nodes are even considered) is still the
    hop-bounded BFS reach in both cases -- only the score assigned to each
    candidate differs.

    Seed retrieval always uses the cheap bi-encoder (has to scan every
    sentence in the passage set). `use_reranker=True` swaps semantic_score
    for the already graph-narrowed candidate pool from bi-encoder cosine
    similarity to a cross-encoder relevance score (see retrieval/reranker.py)
    -- sharper top-of-list precision, which matters most at a tight token
    budget where every selected slot has to count. Default off until
    validated; a pure component swap, so alpha's meaning is unchanged."""
    g = build_graph(item, edge_types=graph_edge_types)
    seed_keys = [s.key for s in seed_retrieve(item, n=seed_n)]
    dist, parent = bfs_hop_distances_with_parents(g, seed_keys)
    within_reach = {k: d for k, d in dist.items() if d <= max_hops}

    sentences_by_key = {s.key: s for s in item.all_sentences()}
    keys = list(within_reach)

    def path_to(key: tuple[str, int]) -> tuple[tuple[str, int], ...]:
        reversed_path = []
        current: tuple[str, int] | None = key
        while current is not None:
            reversed_path.append(current)
            current = parent.get(current)
        return tuple(reversed(reversed_path))

    if use_reranker:
        scores = rerank_scores(item.question, [sentences_by_key[k].embedding_text for k in keys])
        semantic_scores = dict(zip(keys, scores))
        vecs_by_key = {}
    else:
        q_vec = embed(item.question)
        vecs = embed_batch([sentences_by_key[k].embedding_text for k in keys])
        semantic_scores = {k: cosine_sim(q_vec, v) for k, v in zip(keys, vecs)}
        vecs_by_key = dict(zip(keys, vecs))

    if graph_rel_method == "ppr":
        source_weights = None
        if query_weighted_ppr:
            if ppr_seed_temperature <= 0:
                raise ValueError("ppr_seed_temperature must be positive")
            logits = [semantic_scores[k] / ppr_seed_temperature for k in seed_keys]
            peak = max(logits, default=0.0)
            exp_values = [math.exp(value - peak) for value in logits]
            source_weights = dict(zip(seed_keys, exp_values))
        ppr_raw = personalized_pagerank_scores(g, seed_keys, source_weights=source_weights)
        pool_values = [ppr_raw.get(k, 0.0) for k in keys]
        max_val = max(pool_values) if pool_values else 0.0
        graph_scores = {k: (ppr_raw.get(k, 0.0) / max_val if max_val > 0 else 0.0) for k in keys}
    else:
        graph_scores = {k: 1.0 / (1 + within_reach[k]) for k in keys}

    candidates = [
        Candidate(
            sentence=sentences_by_key[k],
            semantic_score=semantic_scores[k],
            graph_score=graph_scores[k],
            hop_distance=within_reach[k],
            embedding=vecs_by_key.get(k),
            path_keys=path_to(k),
            neighbor_keys=tuple(neighbor for neighbor in g.neighbors(k) if neighbor in within_reach),
        )
        for k in keys
    ]
    if conditional_rerank:
        _add_conditional_second_hop_scores(
            item,
            candidates,
            anchor_n=conditional_anchor_n,
            prefer_mentioned_anchors=prefer_mentioned_anchors,
            link_top_l=conditional_link_top_l,
            link_min_score=conditional_link_min_score,
        )
    return candidates


def _add_conditional_second_hop_scores(
    item: QAItem,
    candidates: list[Candidate],
    anchor_n: int = 2,
    prefer_mentioned_anchors: bool = False,
    link_top_l: int | None = 2,
    link_min_score: float | None = None,
) -> None:
    """Score graph-linked second-hop candidates conditioned on strong anchors.

    The original question often retrieves a bridge sentence but is not
    lexically close to the answer-bearing second hop.  This branch augments
    the question with a high-confidence anchor and reranks only cross-passage
    graph neighbours. A link is considered validated only when it is in the
    Top-L conditional scores for its anchor and, when configured, clears the
    minimum score. It is inference-only: no gold supporting facts or answer
    text are consulted.
    """
    if anchor_n <= 0 or not candidates:
        return
    if link_top_l is not None and link_top_l <= 0:
        raise ValueError("link_top_l must be positive or None")
    by_key = {candidate.sentence.key: candidate for candidate in candidates}
    anchors: list[Candidate] = []
    covered_titles: set[str] = set()
    normalized_question = _normalize_title_match(item.question)

    # Multi-hop questions commonly name their source entities explicitly.
    # Prefer one high-CE sentence from each mentioned passage title; this is
    # substantially safer than letting an unrelated but lexically similar
    # passage become the second anchor in a comparison.
    mentioned_titles = (
        {
            candidate.sentence.passage_title
            for candidate in candidates
            if _normalize_title_match(candidate.sentence.passage_title) in normalized_question
        }
        if prefer_mentioned_anchors
        else set()
    )
    for title in sorted(
        mentioned_titles,
        key=lambda value: normalized_question.find(_normalize_title_match(value)),
    ):
        title_candidates = [
            candidate
            for candidate in candidates
            if candidate.sentence.passage_title == title
        ]
        anchor = max(
            title_candidates,
            key=lambda candidate: (
                candidate.semantic_score
                if candidate.semantic_score is not None
                else float("-inf")
            ),
        )
        anchors.append(anchor)
        covered_titles.add(title)
        if len(anchors) >= anchor_n:
            break

    for candidate in sorted(
        candidates,
        key=lambda c: (
            c.semantic_score if c.semantic_score is not None else float("-inf")
        ),
        reverse=True,
    ):
        title = candidate.sentence.passage_title
        if title in covered_titles:
            continue
        anchors.append(candidate)
        covered_titles.add(title)
        if len(anchors) >= anchor_n:
            break

    for anchor in anchors:
        linked = [
            by_key[key]
            for key in (anchor.neighbor_keys or ())
            if key in by_key
            and by_key[key].sentence.passage_title != anchor.sentence.passage_title
        ]
        if not linked:
            continue
        conditioned_question = (
            f"{item.question}\nKnown evidence: {anchor.sentence.embedding_text}"
        )
        scores = rerank_scores(
            conditioned_question,
            [candidate.sentence.embedding_text for candidate in linked],
        )
        ranked_links = sorted(
            zip(linked, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if link_top_l is not None:
            ranked_links = ranked_links[:link_top_l]
        for candidate, score in ranked_links:
            if link_min_score is not None and score < link_min_score:
                continue
            if candidate.conditional_score is None or score > candidate.conditional_score:
                candidate.conditional_score = score
                candidate.conditional_anchor_key = anchor.sentence.key
                candidate.conditional_validated = True


def _normalize_title_match(text: str) -> str:
    """Case/punctuation-insensitive phrase used only for title containment."""
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())
