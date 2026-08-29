from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from rivf.data.schema import Sentence


@dataclass
class Candidate:
    """One retrieval candidate with its component scores kept separate --
    never pre-combined -- so every (alpha, budget) selection downstream is
    cheap re-slicing of an already-computed candidate set, not a re-retrieval."""

    sentence: Sentence
    semantic_score: float | None = None  # Semantic(q, v): BI cosine or CE score
    graph_score: float | None = None  # GraphRel(v): e.g. inverse hop-distance from seed
    hop_distance: int | None = None  # 0 = seed, 1, 2, ...; None where not applicable
    embedding: np.ndarray | None = None  # cached sentence vector, for MMR-style redundancy checks
    path_keys: tuple[tuple[str, int], ...] | None = None  # one shortest seed-to-node path
    neighbor_keys: tuple[tuple[str, int], ...] | None = None  # graph-adjacent candidates
    conditional_score: float | None = None  # CE(q + anchor, v), for second-hop rescue
    conditional_anchor_key: tuple[str, int] | None = None
    conditional_validated: bool = False  # True only for a retained Top-L/threshold link


@dataclass
class RetrievalResult:
    qid: str
    candidates: list[Candidate]  # full pre-pruning set, both scores cached
    selected: list[Sentence]  # this run's actual pruned selection
    context_tokens: int
    latency_s: float


class Retriever(ABC):
    """Every method (baselines, Ours, the HippoRAG adapter) implements this
    one interface, so the experiment runner is fully method-agnostic."""

    @abstractmethod
    def retrieve(
        self, item, alpha: float = 1.0, budget_tokens: int | None = None
    ) -> RetrievalResult: ...
