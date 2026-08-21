from rivf.data.schema import Passage, QAItem, Sentence
from rivf.experiments.ppr_rescue_probe import evaluate_candidates
from rivf.methods.base import Candidate


def _item() -> QAItem:
    sentences = [Sentence("P", i, token) for i, token in enumerate(("a", "b", "c", "d"))]
    return QAItem(
        qid="q",
        question="question",
        answer="answer",
        type="bridge",
        level="hard",
        passages=[Passage("P", sentences)],
        supporting_facts=[("P", 0), ("P", 1)],
    )


def test_ppr_probe_counts_a_gold_fact_rescued_from_semantic_topk():
    item = _item()
    candidates = [
        Candidate(item.passages[0].sentences[0], semantic_score=0.9, graph_score=0.2),
        Candidate(item.passages[0].sentences[1], semantic_score=0.1, graph_score=0.9),
        Candidate(item.passages[0].sentences[2], semantic_score=0.8, graph_score=0.1),
        Candidate(item.passages[0].sentences[3], semantic_score=0.2, graph_score=0.8),
    ]
    result = evaluate_candidates(
        item, candidates, budget_tokens=None, top_k=2, random_trials=20, rng_seed=0
    )
    assert result.semantic_gold == 1
    assert result.recoverable_semantic_misses == 1
    assert result.ppr_rescued_misses == 1
    assert result.union_complete


def test_ppr_probe_excludes_gold_outside_candidate_pool_from_recoverable_misses():
    item = _item()
    candidates = [
        Candidate(item.passages[0].sentences[0], semantic_score=0.9, graph_score=0.9),
        Candidate(item.passages[0].sentences[2], semantic_score=0.8, graph_score=0.8),
    ]
    result = evaluate_candidates(
        item, candidates, budget_tokens=None, top_k=1, random_trials=10, rng_seed=0
    )
    assert result.candidate_gold == 1
    assert result.recoverable_semantic_misses == 0
