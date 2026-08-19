import time

from rivf.data.schema import QAItem
from rivf.eval.efficiency import count_tokens
from rivf.methods.base import RetrievalResult, Retriever
from rivf.retrieval.pruning import prune
from rivf.retrieval.scoring import generate_candidates


class SelectiveGraphRAG(Retriever):
    """Ours: expand -> score -> prune. Orchestrates seed retrieval + graph
    expansion + scoring via `generate_candidates` (semantic and graph score
    computed once, cached separately), then applies `prune` for the
    requested alpha/budget/strategy. This class IS the pipeline -- no
    separate orchestration module."""

    def __init__(self, max_hops: int = 2, strategy: str = "topk", use_reranker: bool = False):
        self.max_hops = max_hops
        self.strategy = strategy
        self.use_reranker = use_reranker

    def retrieve(
        self,
        item: QAItem,
        alpha: float = 0.5,
        budget_tokens: int | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> RetrievalResult:
        start = time.perf_counter()
        candidates = generate_candidates(item, max_hops=self.max_hops, use_reranker=self.use_reranker)
        selected = prune(
            candidates,
            alpha=alpha,
            strategy=self.strategy,
            budget_tokens=budget_tokens,
            top_k=top_k,
            threshold=threshold,
        )
        tokens = sum(count_tokens(s.text) for s in selected)
        return RetrievalResult(
            qid=item.qid,
            candidates=candidates,
            selected=selected,
            context_tokens=tokens,
            latency_s=time.perf_counter() - start,
        )
