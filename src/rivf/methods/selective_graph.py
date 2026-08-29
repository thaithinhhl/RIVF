import time

from rivf.data.schema import QAItem
from rivf.eval.efficiency import count_tokens
from rivf.methods.base import RetrievalResult, Retriever
from rivf.retrieval.pruning import prune
from rivf.retrieval.question_router import DIRECT_COMPARISON, route_question
from rivf.retrieval.scoring import generate_candidates


class SelectiveGraphRAG(Retriever):
    """Reusable GRAFT-v1 retriever for single-item inference.

    The experiment runner provides sweep/caching/provenance orchestration; this
    class mirrors the frozen single operating point for application code.
    """

    def __init__(
        self,
        max_hops: int = 2,
        use_reranker: bool = True,
        anchor_n: int = 2,
        link_top_l: int = 2,
        conditional_weight: float = 0.30,
    ):
        self.max_hops = max_hops
        self.use_reranker = use_reranker
        self.anchor_n = anchor_n
        self.link_top_l = link_top_l
        self.conditional_weight = conditional_weight

    def retrieve(
        self,
        item: QAItem,
        alpha: float = 0.5,
        budget_tokens: int | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> RetrievalResult:
        start = time.perf_counter()
        route = route_question(item.question)
        candidates = generate_candidates(
            item,
            max_hops=self.max_hops,
            use_reranker=self.use_reranker,
            conditional_rerank=route != DIRECT_COMPARISON,
            conditional_anchor_n=self.anchor_n,
            conditional_link_top_l=self.link_top_l,
        )
        selected = prune(
            candidates,
            alpha=1.0,
            strategy=(
                "coverage_topk"
                if route == DIRECT_COMPARISON
                else "graft_pair_topk"
            ),
            budget_tokens=budget_tokens,
            top_k=top_k,
            threshold=threshold,
            conditional_weight=self.conditional_weight,
            preserve_pairs=True,
        )
        tokens = sum(count_tokens(s.text) for s in selected)
        return RetrievalResult(
            qid=item.qid,
            candidates=candidates,
            selected=selected,
            context_tokens=tokens,
            latency_s=time.perf_counter() - start,
        )
