from sentence_transformers import CrossEncoder

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME, device="mps")
    return _model


def rerank_scores(question: str, texts: list[str]) -> list[float]:
    """Cross-encoder relevance score for each (question, text) pair. Unlike
    the bi-encoder semantic_score (independent embeddings, then cosine
    similarity), a cross-encoder processes the pair jointly, giving sharper
    top-of-list precision -- exactly what a tight token budget needs, since
    every slot has to count. No cross-question caching is possible (the
    score is a function of the pair, not of the text alone), so this is only
    applied to the already graph-narrowed candidate pool (~20-30 items),
    not the full ~40-60 raw sentences seed retrieval scans."""
    if not texts:
        return []
    pairs = [[question, t] for t in texts]
    scores = _get_model().predict(pairs, batch_size=len(pairs))
    return [float(s) for s in scores]
