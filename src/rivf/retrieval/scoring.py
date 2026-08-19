from rivf.data.schema import QAItem
from rivf.graph.build import bfs_hop_distances, build_graph, personalized_pagerank_scores
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
    g = build_graph(item)
    seed_keys = [s.key for s in seed_retrieve(item, n=seed_n)]
    dist = bfs_hop_distances(g, seed_keys)
    within_reach = {k: d for k, d in dist.items() if d <= max_hops}

    sentences_by_key = {s.key: s for s in item.all_sentences()}
    keys = list(within_reach)

    if graph_rel_method == "ppr":
        ppr_raw = personalized_pagerank_scores(g, seed_keys)
        pool_values = [ppr_raw.get(k, 0.0) for k in keys]
        max_val = max(pool_values) if pool_values else 0.0
        graph_scores = {k: (ppr_raw.get(k, 0.0) / max_val if max_val > 0 else 0.0) for k in keys}
    else:
        graph_scores = {k: 1.0 / (1 + within_reach[k]) for k in keys}

    if use_reranker:
        scores = rerank_scores(item.question, [sentences_by_key[k].embedding_text for k in keys])
        semantic_scores = dict(zip(keys, scores))
        vecs_by_key = {}
    else:
        q_vec = embed(item.question)
        vecs = embed_batch([sentences_by_key[k].embedding_text for k in keys])
        semantic_scores = {k: cosine_sim(q_vec, v) for k, v in zip(keys, vecs)}
        vecs_by_key = dict(zip(keys, vecs))

    return [
        Candidate(
            sentence=sentences_by_key[k],
            semantic_score=semantic_scores[k],
            graph_score=graph_scores[k],
            hop_distance=within_reach[k],
            embedding=vecs_by_key.get(k),
        )
        for k in keys
    ]
