import itertools
from collections import deque

import networkx as nx

from rivf.data.schema import QAItem, Sentence
from rivf.graph.entities import extract_entities, mentions_title

EDGE_SAME_PASSAGE = "same_passage"
EDGE_ENTITY_OVERLAP = "entity_overlap"
EDGE_TITLE_MENTION = "title_mention"
ALL_EDGE_TYPES = frozenset(
    {EDGE_SAME_PASSAGE, EDGE_ENTITY_OVERLAP, EDGE_TITLE_MENTION}
)

# Title-mention gets the highest weight: HotpotQA bridge questions are built from
# Wikipedia hyperlinks, and anchor text ~= article title, so a sentence naming
# another paragraph's title is usually the *actual* bridge -- more reliable than
# generic entity overlap, which can miss titles that don't tag as a clean NER span.
_EDGE_WEIGHT = {
    EDGE_SAME_PASSAGE: 1.0,
    EDGE_ENTITY_OVERLAP: 1.0,
    EDGE_TITLE_MENTION: 2.0,
}

def build_graph(
    item: QAItem,
    edge_types: set[str] | frozenset[str] | None = None,
) -> nx.Graph:
    """Per-question graph: nodes = every sentence in the 10 distractor
    paragraphs, edges = same-passage adjacency + cross-passage entity overlap
    + cross-passage title mention. Rebuilt fresh per question -- no persistent
    store, since each question's graph is tiny (~40-60 nodes)."""
    enabled = ALL_EDGE_TYPES if edge_types is None else frozenset(edge_types)
    unknown = enabled - ALL_EDGE_TYPES
    if unknown:
        raise ValueError(f"unknown graph edge types: {sorted(unknown)}")

    g = nx.Graph()
    sentences = item.all_sentences()
    for s in sentences:
        g.add_node(s.key, sentence=s)

    if EDGE_SAME_PASSAGE in enabled:
        for passage in item.passages:
            for a, b in itertools.pairwise(passage.sentences):
                _add_edge(g, a.key, b.key, EDGE_SAME_PASSAGE)

    if EDGE_ENTITY_OVERLAP in enabled:
        entities_by_key = {s.key: extract_entities(s.text) for s in sentences}
        for a, b in itertools.combinations(sentences, 2):
            if a.passage_title == b.passage_title:
                continue
            if entities_by_key[a.key] & entities_by_key[b.key]:
                _add_edge(g, a.key, b.key, EDGE_ENTITY_OVERLAP)

    if EDGE_TITLE_MENTION in enabled:
        sentences_by_title: dict[str, list[Sentence]] = {
            p.title: p.sentences for p in item.passages
        }
        for s in sentences:
            for title, title_sentences in sentences_by_title.items():
                if title == s.passage_title:
                    continue
                if mentions_title(s.text, title):
                    for other in title_sentences:
                        _add_edge(g, s.key, other.key, EDGE_TITLE_MENTION)

    return g


def bfs_hop_distances(g: nx.Graph, sources: list[tuple]) -> dict[tuple, int]:
    """Unweighted multi-source BFS: hop distance from the nearest source to
    every node reachable from any of them."""
    dist, _ = bfs_hop_distances_with_parents(g, sources)
    return dist


def bfs_hop_distances_with_parents(
    g: nx.Graph, sources: list[tuple]
) -> tuple[dict[tuple, int], dict[tuple, tuple | None]]:
    """Multi-source BFS plus one deterministic shortest-path parent/node.

    Parent pointers let downstream pruning retain the evidence path when it
    selects an expanded node. The existing distance-only API delegates here
    so candidate reachability and path closure cannot silently diverge.
    """
    dist = {s: 0 for s in sources if s in g}
    parent: dict[tuple, tuple | None] = {s: None for s in dist}
    queue = deque(dist.keys())
    while queue:
        u = queue.popleft()
        for v in g.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    return dist, parent


def personalized_pagerank_scores(
    g: nx.Graph,
    sources: list[tuple],
    source_weights: dict[tuple, float] | None = None,
) -> dict[tuple, float]:
    """GraphRel(v) via Personalized PageRank restarting from the seed nodes
    -- a continuous, multi-path-sensitive alternative to bfs_hop_distances'
    single-shortest-path count. A node reachable by several short paths from
    different seeds accumulates more PPR mass than one reachable by only a
    single path of the same length, which pure hop-distance can't express.

    Raw PPR mass sums to 1 across the WHOLE graph, so values are typically
    tiny (~1/n_nodes) -- callers combining this with a 0-1-scaled semantic
    score should renormalize it (e.g. divide by the max value within the
    candidate pool) rather than using it raw, or it will be numerically
    swamped in any linear blend."""
    valid_sources = [s for s in sources if s in g]
    if not valid_sources:
        return {}
    personalization = dict.fromkeys(g.nodes, 0.0)
    if source_weights is None:
        for s in valid_sources:
            personalization[s] = 1.0 / len(valid_sources)
    else:
        positive = {s: max(0.0, source_weights.get(s, 0.0)) for s in valid_sources}
        total = sum(positive.values())
        if total <= 0:
            for s in valid_sources:
                personalization[s] = 1.0 / len(valid_sources)
        else:
            for s, weight in positive.items():
                personalization[s] = weight / total
    return nx.pagerank(g, alpha=0.85, personalization=personalization, weight="weight")


def _add_edge(g: nx.Graph, u: tuple, v: tuple, kind: str) -> None:
    weight = _EDGE_WEIGHT[kind]
    if g.has_edge(u, v):
        kinds = set(g[u][v].get("kinds", (g[u][v]["kind"],)))
        kinds.add(kind)
        g[u][v]["kinds"] = tuple(sorted(kinds))
        if weight > g[u][v]["weight"]:
            g[u][v]["kind"] = kind
            g[u][v]["weight"] = weight
    else:
        g.add_edge(u, v, kind=kind, kinds=(kind,), weight=weight)
