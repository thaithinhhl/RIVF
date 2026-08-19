import time

from rivf.data.schema import QAItem
from rivf.eval.efficiency import count_tokens
from rivf.methods.base import RetrievalResult, Retriever
from rivf.retrieval.pruning import prune
from rivf.retrieval.scoring import generate_candidates


class FixedHopRAG(Retriever):
    """Graph heuristic baseline: naive expansion to exactly `hops` hops from
    seed. By default (no budget_tokens) nothing is pruned -- this is the
    "full expansion" baseline that shows the raw recall/context trade-off.
    When a budget IS given (Experiment 2), candidates are ranked by semantic
    score within the hop-limited set before the token cutoff, since fixed-hop
    has no other natural internal ranking."""

    def __init__(self, hops: int):
        self.hops = hops

    def retrieve(
        self, item: QAItem, alpha: float = 1.0, budget_tokens: int | None = None
    ) -> RetrievalResult:
        start = time.perf_counter()
        candidates = generate_candidates(item, max_hops=self.hops)
        selected = prune(candidates, alpha=1.0, strategy="topk", budget_tokens=budget_tokens)
        tokens = sum(count_tokens(s.text) for s in selected)
        return RetrievalResult(
            qid=item.qid,
            candidates=candidates,
            selected=selected,
            context_tokens=tokens,
            latency_s=time.perf_counter() - start,
        )
