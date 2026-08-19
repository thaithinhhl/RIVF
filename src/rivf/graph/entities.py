from functools import lru_cache

import spacy

_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _NLP


def normalize_text(entity_text: str) -> str:
    return " ".join(entity_text.lower().split())


@lru_cache(maxsize=None)
def extract_entities(text: str) -> frozenset[str]:
    """Normalized named-entity surface forms in a single sentence, cached per
    unique text so repeated sentences (a passage reused as a distractor across
    many questions) only run NER once."""
    doc = _nlp()(text)
    return frozenset(normalize_text(ent.text) for ent in doc.ents)


def mentions_title(text: str, title: str) -> bool:
    """Whether a sentence's text contains another passage's title as a
    substring (normalized) -- the anchor-text-as-title signal behind
    HotpotQA's Wikipedia-hyperlink bridge questions, which a pure NER
    entity-overlap check can miss (e.g. "Arthur's Magazine" may not tag
    cleanly as a single NER span)."""
    return normalize_text(title) in normalize_text(text)
