import torch
from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_model: CrossEncoder | None = None
_score_cache: dict[tuple[str, str], float] = {}


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _model = CrossEncoder(MODEL_NAME, device=device)
    return _model


def rerank_scores(question: str, texts: list[str]) -> list[float]:
    """Cross-encoder relevance score for each (question, text) pair. Unlike
    the bi-encoder semantic_score (independent embeddings, then cosine
    similarity), a cross-encoder processes the pair jointly, giving sharper
    top-of-list precision -- exactly what a tight token budget needs, since
    every slot has to count. No cross-question caching is possible (the
    score is a function of the pair, not of the text alone). An in-process
    pair cache avoids recomputing the same scores when controlled baselines
    rerank overlapping candidate pools for one question. The cache is not
    persisted across experiment processes."""
    if not texts:
        return []
    missing = list(dict.fromkeys(t for t in texts if (question, t) not in _score_cache))
    if missing:
        pairs = [[question, text] for text in missing]
        scores = _get_model().predict(pairs, batch_size=len(pairs))
        _score_cache.update({(question, text): float(score) for text, score in zip(missing, scores)})
    return [_score_cache[(question, text)] for text in texts]
