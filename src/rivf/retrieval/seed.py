from rivf.data.schema import QAItem, Sentence
from rivf.retrieval.embeddings import cosine_sim, embed, embed_batch

DEFAULT_SEED_N = 5


def seed_retrieve(item: QAItem, n: int = DEFAULT_SEED_N) -> list[Sentence]:
    """Top-n sentences across all 10 paragraphs by embedding similarity to the
    question -- this is the dense/semantic seed retrieval step every method
    starts from (including Dense RAG itself, which just skips the graph step
    that follows)."""
    sentences = item.all_sentences()
    q_vec = embed(item.question)
    sent_vecs = embed_batch([s.embedding_text for s in sentences])
    scored = sorted(
        zip(sentences, sent_vecs), key=lambda pair: cosine_sim(q_vec, pair[1]), reverse=True
    )
    return [s for s, _ in scored[:n]]
