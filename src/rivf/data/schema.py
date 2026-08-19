from dataclasses import dataclass


@dataclass(frozen=True)
class Sentence:
    passage_title: str
    sent_id: int
    text: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.passage_title, self.sent_id)

    @property
    def embedding_text(self) -> str:
        """Title-prefixed text for embedding only (never for token counting
        or the LLM prompt, which already groups sentences under their title
        separately) -- a bare sentence like "He was born in 1965" is
        under-specified alone; pairing it with its title gives the embedding
        model the referential context to place it correctly."""
        return f"{self.passage_title}: {self.text}"


@dataclass(frozen=True)
class Passage:
    title: str
    sentences: list[Sentence]


@dataclass(frozen=True)
class QAItem:
    qid: str
    question: str
    answer: str
    type: str
    level: str
    passages: list[Passage]
    supporting_facts: list[tuple[str, int]]

    def all_sentences(self) -> list[Sentence]:
        return [s for p in self.passages for s in p.sentences]
