import numpy as np

from rivf.data.schema import Passage, QAItem, Sentence
from rivf.methods.base import Candidate
from rivf.retrieval.pruning import combined_score, normalized_combined_scores, prune
from rivf.retrieval.scoring import _add_conditional_second_hop_scores


def _candidate(title: str, sent_id: int, text: str, semantic: float, graph: float, hop: int) -> Candidate:
    return Candidate(
        sentence=Sentence(passage_title=title, sent_id=sent_id, text=text),
        semantic_score=semantic,
        graph_score=graph,
        hop_distance=hop,
    )


def _candidates():
    return [
        _candidate("P", 0, "aaaa", semantic=0.9, graph=0.2, hop=2),  # high semantic, low graph
        _candidate("P", 1, "bb", semantic=0.1, graph=1.0, hop=0),  # low semantic, high graph (seed)
        _candidate("Q", 0, "cc", semantic=0.5, graph=0.5, hop=1),  # middle of both
    ]


def test_combined_score_is_weighted_average():
    c = _candidate("P", 0, "x", semantic=0.8, graph=0.2, hop=1)
    assert combined_score(c, alpha=1.0) == 0.8
    assert combined_score(c, alpha=0.0) == 0.2
    assert round(combined_score(c, alpha=0.5), 4) == 0.5


def test_topk_alpha_1_orders_by_semantic_only():
    selected = prune(_candidates(), alpha=1.0, strategy="topk")
    assert [s.sent_id for s in selected] == [0, 0, 1]  # P0(0.9) > Q0(0.5) > P1(0.1), by (title,id) here P0 first
    assert selected[0].passage_title == "P" and selected[0].sent_id == 0


def test_topk_alpha_0_orders_by_graph_only():
    selected = prune(_candidates(), alpha=0.0, strategy="topk")
    assert selected[0].passage_title == "P" and selected[0].sent_id == 1  # graph=1.0, the seed


def test_topk_is_deterministic_across_repeated_calls():
    candidates = _candidates()
    first = prune(candidates, alpha=0.5, strategy="topk")
    second = prune(candidates, alpha=0.5, strategy="topk")
    assert [s.key for s in first] == [s.key for s in second]


def test_random_strategy_is_deterministic_given_fixed_seed():
    candidates = _candidates()
    first = prune(candidates, strategy="random", rng_seed=42)
    second = prune(candidates, strategy="random", rng_seed=42)
    assert [s.key for s in first] == [s.key for s in second]


def test_budget_tokens_stops_before_exceeding_budget():
    candidates = _candidates()  # texts: "aaaa"(P0), "bb"(P1), "cc"(Q0); ranked by alpha=1.0: P0, Q0, P1
    # token counts are small integers per short string; find a budget that
    # admits exactly the top-ranked candidate and excludes the rest.
    from rivf.eval.efficiency import count_tokens

    first_cost = count_tokens("aaaa")
    selected = prune(candidates, alpha=1.0, strategy="topk", budget_tokens=first_cost)
    assert len(selected) == 1
    assert selected[0].passage_title == "P" and selected[0].sent_id == 0


def test_top_k_limits_count_regardless_of_tokens():
    selected = prune(_candidates(), alpha=1.0, strategy="topk", top_k=2)
    assert len(selected) == 2


def test_threshold_keeps_only_candidates_at_or_above_cutoff():
    selected = prune(_candidates(), alpha=1.0, strategy="threshold", threshold=0.5)
    kept = {(s.passage_title, s.sent_id) for s in selected}
    assert kept == {("P", 0), ("Q", 0)}  # semantic scores 0.9 and 0.5 both >= 0.5; 0.1 excluded


def test_no_budget_and_no_top_k_keeps_everything():
    selected = prune(_candidates(), alpha=1.0, strategy="topk")
    assert len(selected) == 3


def test_rrf_alpha_1_orders_by_semantic_rank_only():
    selected = prune(_candidates(), alpha=1.0, strategy="rrf")
    assert [(s.passage_title, s.sent_id) for s in selected] == [("P", 0), ("Q", 0), ("P", 1)]


def test_rrf_alpha_0_orders_by_graph_rank_only():
    selected = prune(_candidates(), alpha=0.0, strategy="rrf")
    assert [(s.passage_title, s.sent_id) for s in selected] == [("P", 1), ("Q", 0), ("P", 0)]


def test_rrf_is_deterministic_across_repeated_calls():
    candidates = _candidates()
    first = prune(candidates, alpha=0.5, strategy="rrf")
    second = prune(candidates, alpha=0.5, strategy="rrf")
    assert [s.key for s in first] == [s.key for s in second]


def test_rrf_is_not_dominated_by_an_unnormalized_outlier_value_unlike_topk():
    # An outlier with a huge, out-of-scale semantic_score but the WORST
    # graph rank can dominate a raw weighted sum (combined_score) purely by
    # magnitude. RRF only sees rank position, so a candidate that's merely
    # "good on both" can beat an outlier that's "extreme on one, worst on
    # the other" -- this is the scale-mismatch failure mode RRF avoids.
    outlier = _candidate("OUT", 0, "outlier", semantic=1000.0, graph=0.0, hop=9)  # rank: semantic=1st, graph=last
    balanced = _candidate("BAL", 0, "balanced", semantic=0.5, graph=0.9, hop=0)  # rank: semantic=2nd, graph=1st
    filler = _candidate("FIL", 0, "filler", semantic=0.1, graph=0.1, hop=5)  # rank: semantic=last, graph=2nd

    by_topk = prune([outlier, balanced, filler], alpha=0.5, strategy="topk")
    assert by_topk[0].passage_title == "OUT"  # raw magnitude lets the outlier win

    by_rrf = prune([outlier, balanced, filler], alpha=0.5, strategy="rrf")
    assert by_rrf[0].passage_title == "BAL"  # rank-based fusion is not fooled by scale


def _candidate_with_embedding(title: str, score: float, embedding: list[float]) -> Candidate:
    return Candidate(
        sentence=Sentence(passage_title=title, sent_id=0, text=title),
        semantic_score=score,
        graph_score=score,
        embedding=np.array(embedding, dtype=float),
    )


def test_mmr_avoids_picking_a_near_duplicate_of_an_already_selected_candidate():
    # A and B are near-identical in content (same embedding direction) and
    # both score higher than C, which covers different content entirely.
    # Plain topk picks A then B (redundant); MMR should prefer diverse C
    # over redundant B once A is already selected.
    a = _candidate_with_embedding("A", score=0.9, embedding=[1.0, 0.0])
    b = _candidate_with_embedding("B", score=0.85, embedding=[1.0, 0.0])
    c = _candidate_with_embedding("C", score=0.7, embedding=[0.0, 1.0])

    by_topk = prune([a, b, c], alpha=0.5, strategy="topk", top_k=2)
    assert {s.passage_title for s in by_topk} == {"A", "B"}

    by_mmr = prune([a, b, c], alpha=0.5, strategy="mmr", top_k=2)
    assert {s.passage_title for s in by_mmr} == {"A", "C"}


def test_mmr_first_pick_is_still_just_the_highest_scoring_candidate():
    a = _candidate_with_embedding("A", score=0.9, embedding=[1.0, 0.0])
    b = _candidate_with_embedding("B", score=0.5, embedding=[0.0, 1.0])
    selected = prune([a, b], alpha=0.5, strategy="mmr", top_k=1)
    assert selected[0].passage_title == "A"


def test_mmr_is_deterministic_across_repeated_calls():
    a = _candidate_with_embedding("A", score=0.9, embedding=[1.0, 0.0])
    b = _candidate_with_embedding("B", score=0.85, embedding=[1.0, 0.0])
    c = _candidate_with_embedding("C", score=0.7, embedding=[0.0, 1.0])
    first = prune([a, b, c], alpha=0.5, strategy="mmr", top_k=2)
    second = prune([a, b, c], alpha=0.5, strategy="mmr", top_k=2)
    assert [s.key for s in first] == [s.key for s in second]


def test_mmr_falls_back_to_relevance_only_when_embeddings_missing():
    # Candidates without an embedding (e.g. synthetic/test data) shouldn't
    # crash MMR -- redundancy is just treated as 0 for them.
    a = _candidate("P", 0, "x", semantic=0.9, graph=0.9, hop=0)
    b = _candidate("Q", 0, "y", semantic=0.5, graph=0.5, hop=1)
    selected = prune([a, b], alpha=0.5, strategy="mmr", top_k=2)
    assert len(selected) == 2


def test_interleave_alternates_between_semantic_and_graph_rankings():
    # sem order: P0(0.9), Q0(0.5), P1(0.1); graph order: P1(1.0), Q0(0.5), P0(0.2)
    selected = prune(_candidates(), strategy="interleave", top_k=3)
    assert [(s.passage_title, s.sent_id) for s in selected] == [("P", 0), ("P", 1), ("Q", 0)]


def test_interleave_first_pick_is_top_semantic():
    selected = prune(_candidates(), strategy="interleave", top_k=1)
    assert selected[0].passage_title == "P" and selected[0].sent_id == 0


def test_interleave_dedups_when_both_channels_agree_on_top_pick():
    # A is the top candidate on both rankings; the graph channel's turn must
    # skip past it (already selected) and fall through to B, not repeat A.
    a = _candidate("A", 0, "x", semantic=0.9, graph=1.0, hop=0)
    b = _candidate("B", 0, "y", semantic=0.5, graph=0.5, hop=1)
    selected = prune([a, b], strategy="interleave", top_k=2)
    assert [(s.passage_title, s.sent_id) for s in selected] == [("A", 0), ("B", 0)]


def test_interleave_skips_an_over_budget_pick_without_stopping():
    # A is expensive and would be the semantic channel's first pick; skipping
    # it on overflow should still let both cheap picks (B via graph, C via
    # semantic) through, unlike "topk"'s stop-at-first-overflow cutoff.
    expensive = _candidate("A", 0, "word " * 50, semantic=0.9, graph=0.1, hop=2)
    cheap_graph = _candidate("B", 0, "b", semantic=0.1, graph=1.0, hop=0)
    cheap_semantic = _candidate("C", 0, "c", semantic=0.8, graph=0.05, hop=2)
    selected = prune(
        [expensive, cheap_graph, cheap_semantic], strategy="interleave", budget_tokens=5
    )
    titles = {s.passage_title for s in selected}
    assert titles == {"B", "C"}


def test_interleave_is_deterministic_across_repeated_calls():
    cands = _candidates()
    first = prune(cands, strategy="interleave", top_k=3)
    second = prune(cands, strategy="interleave", top_k=3)
    assert [s.key for s in first] == [s.key for s in second]


def test_normalized_topk_removes_raw_score_scale_domination():
    outlier = _candidate("OUT", 0, "outlier", semantic=1000.0, graph=0.0, hop=2)
    balanced = _candidate("BAL", 0, "balanced", semantic=0.5, graph=0.9, hop=1)
    filler = _candidate("FIL", 0, "filler", semantic=0.1, graph=0.1, hop=0)
    scores = normalized_combined_scores([outlier, balanced, filler], alpha=0.5)
    assert scores[1] > scores[0]
    selected = prune(
        [outlier, balanced, filler], alpha=0.5, strategy="normalized_topk", top_k=1
    )
    assert selected[0].passage_title == "BAL"


def test_adaptive_mass_can_stop_before_filling_the_hard_budget():
    candidates = _candidates()
    selected = prune(
        candidates,
        alpha=1.0,
        strategy="adaptive_mass",
        budget_tokens=300,
        mass_threshold=0.7,
        mass_temperature=0.05,
        min_nodes=1,
    )
    assert len(selected) < len(candidates)
    assert selected[0].key == ("P", 0)


def test_adaptive_mass_path_closure_keeps_intermediate_nodes():
    a = _candidate("P", 0, "a", semantic=0.1, graph=0.1, hop=0)
    b = _candidate("P", 1, "b", semantic=0.2, graph=0.2, hop=1)
    c = _candidate("Q", 0, "c", semantic=0.9, graph=0.9, hop=2)
    a.path_keys = (a.sentence.key,)
    b.path_keys = (a.sentence.key, b.sentence.key)
    c.path_keys = (a.sentence.key, b.sentence.key, c.sentence.key)
    selected = prune(
        [a, b, c],
        alpha=1.0,
        strategy="adaptive_mass",
        budget_tokens=300,
        mass_threshold=0.5,
        mass_temperature=0.05,
        min_nodes=1,
        path_closure=True,
    )
    assert {sentence.key for sentence in selected} == {a.sentence.key, b.sentence.key, c.sentence.key}


def test_coverage_topk_rewards_an_unseen_passage():
    a = _candidate("A", 0, "a", semantic=0.9, graph=0.0, hop=0)
    redundant = _candidate("A", 1, "b", semantic=0.8, graph=0.0, hop=0)
    other_hop = _candidate("B", 0, "c", semantic=0.7, graph=0.0, hop=1)
    selected = prune(
        [a, redundant, other_hop],
        alpha=1.0,
        strategy="coverage_topk",
        top_k=2,
        coverage_bonus=0.6,
    )
    assert [sentence.key for sentence in selected] == [a.sentence.key, other_hop.sentence.key]


def test_coverage_topk_skips_an_overflowing_candidate_and_fills_budget():
    from rivf.eval.efficiency import count_tokens

    first = _candidate("A", 0, "a", semantic=0.9, graph=0.0, hop=0)
    expensive = _candidate("B", 0, "word " * 50, semantic=0.8, graph=0.0, hop=1)
    cheap = _candidate("C", 0, "c", semantic=0.7, graph=0.0, hop=1)
    budget = count_tokens(first.sentence.text) + count_tokens(cheap.sentence.text)
    selected = prune(
        [first, expensive, cheap],
        alpha=1.0,
        strategy="coverage_topk",
        budget_tokens=budget,
    )
    assert [sentence.key for sentence in selected] == [first.sentence.key, cheap.sentence.key]


def test_conditional_topk_retains_rescued_node_and_its_anchor():
    anchor = _candidate("A", 0, "anchor", semantic=0.9, graph=0.0, hop=0)
    ordinary = _candidate("B", 0, "ordinary", semantic=0.8, graph=0.0, hop=1)
    rescued = _candidate("C", 0, "rescued", semantic=0.1, graph=0.0, hop=1)
    rescued.conditional_score = 1.0
    rescued.conditional_anchor_key = anchor.sentence.key
    selected = prune(
        [anchor, ordinary, rescued],
        alpha=1.0,
        strategy="conditional_topk",
        top_k=2,
        conditional_weight=1.0,
    )
    assert {sentence.key for sentence in selected} == {
        anchor.sentence.key,
        rescued.sentence.key,
    }


def test_conditional_topk_without_conditional_scores_matches_relevance_order():
    selected = prune(
        _candidates(),
        alpha=1.0,
        strategy="conditional_topk",
        top_k=2,
        conditional_weight=0.5,
    )
    assert [sentence.key for sentence in selected] == [("P", 0), ("Q", 0)]


def test_graph_rescue_promotes_only_graph_topk_semantic_miss_with_conditional_support():
    anchor = _candidate("A", 0, "anchor", semantic=1.0, graph=0.2, hop=0)
    semantic_second = _candidate("B", 0, "semantic", semantic=0.9, graph=0.1, hop=1)
    rescued = _candidate("C", 0, "rescued", semantic=0.5, graph=1.0, hop=1)
    graph_only = _candidate("D", 0, "unvalidated", semantic=0.1, graph=0.8, hop=1)
    rescued.conditional_score = 1.0
    rescued.conditional_anchor_key = anchor.sentence.key

    without_rescue = prune(
        [anchor, semantic_second, rescued, graph_only],
        alpha=1.0,
        strategy="graph_rescue_topk",
        top_k=2,
        conditional_weight=0.0,
        graph_rescue_weight=0.0,
        graph_rescue_k=2,
    )
    assert [sentence.key for sentence in without_rescue] == [
        anchor.sentence.key,
        semantic_second.sentence.key,
    ]

    with_rescue = prune(
        [anchor, semantic_second, rescued, graph_only],
        alpha=1.0,
        strategy="graph_rescue_topk",
        top_k=2,
        conditional_weight=0.0,
        graph_rescue_weight=1.0,
        graph_rescue_k=2,
    )
    assert {sentence.key for sentence in with_rescue} == {
        anchor.sentence.key,
        rescued.sentence.key,
    }


def test_chain_set_topk_reserves_a_pair_for_each_anchor():
    a1 = _candidate("A", 0, "a1", semantic=0.9, graph=0.0, hop=0)
    a2 = _candidate("B", 0, "a2", semantic=0.8, graph=0.0, hop=0)
    e1 = _candidate("C", 0, "e1", semantic=0.4, graph=0.0, hop=1)
    e2 = _candidate("D", 0, "e2", semantic=0.3, graph=0.0, hop=1)
    e1.conditional_score = 0.9
    e1.conditional_anchor_key = a1.sentence.key
    e2.conditional_score = 0.8
    e2.conditional_anchor_key = a2.sentence.key
    selected = prune(
        [a1, a2, e1, e2],
        alpha=1.0,
        strategy="chain_set_topk",
        top_k=4,
        conditional_weight=0.3,
    )
    assert {sentence.key for sentence in selected} == {
        a1.sentence.key,
        a2.sentence.key,
        e1.sentence.key,
        e2.sentence.key,
    }


def test_conditional_rerank_prefers_titles_mentioned_in_question(monkeypatch):
    film_a = _candidate("Film A", 0, "directed by Person A", semantic=0.8, graph=0.0, hop=0)
    film_b = _candidate("Film B", 0, "directed by Person B", semantic=0.7, graph=0.0, hop=0)
    distractor = _candidate("Unrelated", 0, "film director", semantic=0.99, graph=0.0, hop=0)
    person_a = _candidate("Person A", 0, "born 1900", semantic=0.2, graph=0.0, hop=1)
    person_b = _candidate("Person B", 0, "born 1910", semantic=0.2, graph=0.0, hop=1)
    film_a.neighbor_keys = (person_a.sentence.key,)
    film_b.neighbor_keys = (person_b.sentence.key,)
    distractor.neighbor_keys = ()
    item = QAItem(
        qid="q",
        question="Which film has the director born first, Film A or Film B?",
        answer="Film A",
        type="",
        level="",
        passages=[
            Passage(title=c.sentence.passage_title, sentences=[c.sentence])
            for c in [film_a, film_b, distractor, person_a, person_b]
        ],
        supporting_facts=[],
    )
    monkeypatch.setattr(
        "rivf.retrieval.scoring.rerank_scores",
        lambda question, texts: [1.0] * len(texts),
    )
    candidates = [film_a, film_b, distractor, person_a, person_b]
    _add_conditional_second_hop_scores(
        item, candidates, anchor_n=2, prefer_mentioned_anchors=True
    )
    assert person_a.conditional_anchor_key == film_a.sentence.key
    assert person_b.conditional_anchor_key == film_b.sentence.key
