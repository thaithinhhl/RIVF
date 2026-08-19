from rivf.data.schema import Sentence
from rivf.methods.base import Candidate
from rivf.retrieval.adaptive_alpha import LOW_ALPHA, estimate_alpha


def _candidate(title: str, sent_id: int, text: str, semantic: float, graph: float, hop: int) -> Candidate:
    return Candidate(
        sentence=Sentence(passage_title=title, sent_id=sent_id, text=text),
        semantic_score=semantic,
        graph_score=graph,
        hop_distance=hop,
    )


def test_estimate_alpha_returns_base_alpha_when_signals_agree_and_confident():
    # Semantic and graph rankings pick the same top candidates (high jaccard),
    # and semantic confidence is high -- nothing suggests semantic is blind.
    candidates = [
        _candidate("P", 0, "x", semantic=0.95, graph=1.0, hop=0),
        _candidate("P", 1, "y", semantic=0.9, graph=0.5, hop=1),
    ]
    assert estimate_alpha(candidates, base_alpha=0.6, budget_tokens=300) == 0.6


def test_estimate_alpha_returns_low_alpha_when_semantic_is_unconfident_and_disagrees_with_graph():
    # A tight budget (fits only one sentence) forces semantic-only and
    # graph-only to each pick a single, DIFFERENT candidate: semantic (all
    # scores low, so low confidence) prefers Q0; graph prefers the seed P0,
    # which semantic then "misses" entirely -- the semantic-blind pattern.
    candidates = [
        _candidate("P", 0, "x", semantic=0.02, graph=1.0, hop=0),  # seed; graph's top pick
        _candidate("Q", 0, "y", semantic=0.05, graph=0.5, hop=1),  # semantic's (weak) top pick
        _candidate("R", 0, "z", semantic=0.01, graph=0.333, hop=2),
    ]
    assert estimate_alpha(candidates, base_alpha=0.6, budget_tokens=1) == LOW_ALPHA


def test_estimate_alpha_handles_empty_candidates():
    assert estimate_alpha([], base_alpha=0.6) == 0.6
