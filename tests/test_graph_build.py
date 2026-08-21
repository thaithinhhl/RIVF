from pathlib import Path

import networkx as nx
import pytest

from rivf.data.hotpotqa import load_hotpotqa
from rivf.graph.build import (
    EDGE_ENTITY_OVERLAP,
    EDGE_SAME_PASSAGE,
    EDGE_TITLE_MENTION,
    build_graph,
    personalized_pagerank_scores,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_hotpot_question.json"


@pytest.fixture
def item():
    return load_hotpotqa(FIXTURE)[0]


def test_loader_parses_fixture_correctly(item):
    assert item.qid == "fixture-0001"
    assert [p.title for p in item.passages] == ["Alpha Bridge", "John Smith", "Random Topic"]
    assert item.supporting_facts == [("Alpha Bridge", 0), ("John Smith", 0)]
    assert len(item.all_sentences()) == 6


def test_same_passage_adjacency_edge(item):
    g = build_graph(item)
    assert g.has_edge(("Alpha Bridge", 0), ("Alpha Bridge", 1))
    assert g[("Alpha Bridge", 0)][("Alpha Bridge", 1)]["kind"] == EDGE_SAME_PASSAGE


def test_title_mention_bridges_alpha_bridge_to_john_smith(item):
    # "Alpha Bridge is a movie directed by John Smith." mentions the title of
    # the "John Smith" passage -- this is the bridge edge the question depends on.
    g = build_graph(item)
    assert g.has_edge(("Alpha Bridge", 0), ("John Smith", 0))
    assert g[("Alpha Bridge", 0)][("John Smith", 0)]["kind"] == EDGE_TITLE_MENTION
    assert g.has_edge(("Alpha Bridge", 0), ("John Smith", 1))


def test_cross_passage_entity_overlap_on_shared_non_title_entity(item):
    # both passages separately mention "Toronto" (a GPE, not anyone's title)
    g = build_graph(item)
    assert g.has_edge(("Alpha Bridge", 1), ("John Smith", 1))
    assert g[("Alpha Bridge", 1)][("John Smith", 1)]["kind"] == EDGE_ENTITY_OVERLAP


def test_distractor_passage_is_isolated(item):
    # "Random Topic" shares no entities and no title mentions with the other
    # two passages -- the graph should correctly NOT connect it to anything.
    g = build_graph(item)
    for sent_id in (0, 1):
        assert g.degree[("Random Topic", sent_id)] == 1  # only its own same-passage neighbor
    assert not g.has_edge(("Random Topic", 0), ("Alpha Bridge", 0))
    assert not g.has_edge(("Random Topic", 0), ("John Smith", 0))


def test_gold_supporting_facts_are_within_two_hops_via_bridge(item):
    import networkx as nx

    g = build_graph(item)
    seed = ("Alpha Bridge", 0)
    dist = nx.shortest_path_length(g, source=seed)
    for title, sent_id in item.supporting_facts:
        assert dist[(title, sent_id)] <= 2


def test_ppr_gives_the_seed_itself_a_high_score(item):
    g = build_graph(item)
    seed = ("Alpha Bridge", 0)
    scores = personalized_pagerank_scores(g, [seed])
    # the seed restarts the walk here every time it resets, so it should
    # score at least as high as any node reachable only by leaving it
    assert scores[seed] >= max(v for k, v in scores.items() if k != seed)


def test_ppr_gives_the_isolated_distractor_passage_near_zero_score(item):
    g = build_graph(item)
    seed = ("Alpha Bridge", 0)
    scores = personalized_pagerank_scores(g, [seed])
    # "Random Topic" shares no edges with anything the seed can reach
    assert scores[("Random Topic", 0)] < 0.01


def test_ppr_returns_empty_when_no_valid_seed_in_graph(item):
    g = build_graph(item)
    assert personalized_pagerank_scores(g, [("Nonexistent", 99)]) == {}


def test_ppr_can_weight_semantically_stronger_seed_more_heavily():
    g = nx.Graph()
    g.add_edge("strong", "middle", weight=1.0)
    g.add_edge("middle", "weak", weight=1.0)
    scores = personalized_pagerank_scores(
        g,
        ["strong", "weak"],
        source_weights={"strong": 0.95, "weak": 0.05},
    )
    assert scores["strong"] > scores["weak"]
