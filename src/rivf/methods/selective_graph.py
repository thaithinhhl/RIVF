import time

from rivf.data.schema import QAItem
from rivf.eval.efficiency import count_tokens
from rivf.methods.base import Candidate, RetrievalResult, Retriever
from rivf.retrieval.embeddings import cosine_sim, embed, embed_batch
from rivf.retrieval.pruning import prune
from rivf.retrieval.question_router import (
    BRIDGE_COMPARISON,
    DIRECT_COMPARISON,
    route_question,
    route_question_v2,
)
from rivf.retrieval.reranker import rerank_scores
from rivf.retrieval.scoring import generate_candidates, question_mentioned_titles


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


class GraftV2RAG(Retriever):
    """Route-adaptive, minimal-witness GRAFT-v2 retriever.

    Direct comparisons bypass graph expansion and reserve evidence for titles
    named in the question. Chained routes retain graph proposal but require a
    non-negative conditional gain before an anchor-endpoint link is admitted,
    then select only the complete witness chains required by the route.
    """

    def __init__(
        self,
        max_hops: int = 2,
        use_reranker: bool = True,
        anchor_n: int = 2,
        link_top_l: int = 2,
        conditional_weight: float = 0.30,
        conditional_link_min_gain: float = 0.0,
        token_cost_power: float = 1.0,
        min_marginal_score: float = 0.015,
        marginal_coverage_bonus: float = 0.0,
        witness_rescue_max_units: int = 0,
    ):
        self.max_hops = max_hops
        self.use_reranker = use_reranker
        self.anchor_n = anchor_n
        self.link_top_l = link_top_l
        self.conditional_weight = conditional_weight
        self.conditional_link_min_gain = conditional_link_min_gain
        self.token_cost_power = token_cost_power
        self.min_marginal_score = min_marginal_score
        self.marginal_coverage_bonus = marginal_coverage_bonus
        self.witness_rescue_max_units = witness_rescue_max_units

    def _direct_candidates(self, item: QAItem) -> list[Candidate]:
        sentences = item.all_sentences()
        if self.use_reranker:
            scores = rerank_scores(
                item.question, [sentence.embedding_text for sentence in sentences]
            )
            return [
                Candidate(sentence=sentence, semantic_score=score)
                for sentence, score in zip(sentences, scores)
            ]
        question_vector = embed(item.question)
        vectors = embed_batch([sentence.embedding_text for sentence in sentences])
        return [
            Candidate(
                sentence=sentence,
                semantic_score=cosine_sim(question_vector, vector),
                embedding=vector,
            )
            for sentence, vector in zip(sentences, vectors)
        ]

    def retrieve(
        self,
        item: QAItem,
        alpha: float = 1.0,
        budget_tokens: int | None = None,
    ) -> RetrievalResult:
        start = time.perf_counter()
        route = route_question_v2(item.question)
        if route == DIRECT_COMPARISON:
            candidates = self._direct_candidates(item)
            preferred_titles = question_mentioned_titles(
                item.question,
                {candidate.sentence.passage_title for candidate in candidates},
            )
            selected = prune(
                candidates,
                strategy="entity_witness",
                budget_tokens=budget_tokens,
                preferred_titles=preferred_titles,
                token_cost_power=self.token_cost_power,
                min_marginal_score=self.min_marginal_score,
                witness_min_titles=2,
                witness_rescue_max_units=self.witness_rescue_max_units,
                serialized_cost_aware=True,
            )
        else:
            candidates = generate_candidates(
                item,
                max_hops=self.max_hops,
                use_reranker=self.use_reranker,
                conditional_rerank=True,
                conditional_anchor_n=self.anchor_n,
                prefer_mentioned_anchors=route == BRIDGE_COMPARISON,
                conditional_link_top_l=self.link_top_l,
                conditional_link_min_gain=self.conditional_link_min_gain,
            )
            selected = prune(
                candidates,
                strategy="graft_v2_witness",
                budget_tokens=budget_tokens,
                conditional_weight=self.conditional_weight,
                preferred_titles=question_mentioned_titles(
                    item.question,
                    {candidate.sentence.passage_title for candidate in candidates},
                ) if route == BRIDGE_COMPARISON else (),
                token_cost_power=self.token_cost_power,
                min_marginal_score=self.min_marginal_score,
                marginal_coverage_bonus=self.marginal_coverage_bonus,
                witness_chain_target=2 if route == BRIDGE_COMPARISON else 1,
                witness_fallback_min_nodes=4 if route == BRIDGE_COMPARISON else 2,
                witness_rescue_max_units=self.witness_rescue_max_units,
                serialized_cost_aware=True,
            )
        tokens = sum(count_tokens(sentence.text) for sentence in selected)
        return RetrievalResult(
            qid=item.qid,
            candidates=candidates,
            selected=selected,
            context_tokens=tokens,
            latency_s=time.perf_counter() - start,
        )
