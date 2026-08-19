import time

from rivf.data.schema import QAItem
from rivf.eval.efficiency import count_tokens
from rivf.methods.base import Candidate, RetrievalResult, Retriever
from rivf.retrieval.embeddings import cosine_sim, embed, embed_batch
from rivf.retrieval.pruning import prune


class DenseRAG(Retriever):
    """Control baseline: pure semantic retrieval over every sentence in the
    10 paragraphs. No graph involved at all."""

    def retrieve(
        self, item: QAItem, alpha: float = 1.0, budget_tokens: int | None = None
    ) -> RetrievalResult:
        start = time.perf_counter()
        sentences = item.all_sentences()
        q_vec = embed(item.question)
        vecs = embed_batch([s.embedding_text for s in sentences])
        candidates = [
            Candidate(sentence=s, semantic_score=cosine_sim(q_vec, v), embedding=v)
            for s, v in zip(sentences, vecs)
        ]
        selected = prune(candidates, alpha=1.0, strategy="topk", budget_tokens=budget_tokens)
        tokens = sum(count_tokens(s.text) for s in selected)
        return RetrievalResult(
            qid=item.qid,
            candidates=candidates,
            selected=selected,
            context_tokens=tokens,
            latency_s=time.perf_counter() - start,
        )
