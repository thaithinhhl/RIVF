from functools import lru_cache
import re
import unicodedata

import spacy

_NLP = None
_ALIAS_STOPWORDS = frozenset({
    "a", "an", "and", "director", "film", "of", "saint", "st", "the",
})


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _NLP


def normalize_text(entity_text: str) -> str:
    return " ".join(entity_text.lower().split())


def canonical_entity_key(entity_text: str) -> str:
    """Alias key for entity transport, not for lexical evidence matching.

    Wikipedia disambiguators, accents, punctuation, and spacing variants such
    as ``Roy Mack (director)``/``Roy Mack`` or
    ``Andre DeToth``/``André De Toth`` map to the same key.
    """
    without_disambiguator = re.sub(r"\s*\([^()]*\)\s*$", "", entity_text)
    without_honorific = re.sub(
        r"^\s*(?:king|queen|prince|princess|dr\.?|sir)\s+",
        "",
        without_disambiguator,
        flags=re.IGNORECASE,
    )
    decomposed = unicodedata.normalize("NFKD", without_honorific.casefold())
    ascii_like = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return "".join(character for character in ascii_like if character.isalnum())


def _entity_alias_tokens(entity_text: str) -> frozenset[str]:
    without_disambiguator = re.sub(r"\s*\([^()]*\)\s*$", "", entity_text)
    decomposed = unicodedata.normalize("NFKD", without_disambiguator.casefold())
    ascii_like = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return frozenset(
        token
        for token in re.findall(r"[a-z0-9]+", ascii_like)
        if token not in _ALIAS_STOPWORDS
        and token not in {"king", "queen", "prince", "princess", "sultan"}
    )


def entity_alias_match(left: str, right: str) -> bool:
    """Conservative entity-title match for surface aliases in QA passages."""
    if canonical_entity_key(left) == canonical_entity_key(right):
        return True
    shared = _entity_alias_tokens(left) & _entity_alias_tokens(right)
    return len(shared) >= 2 or any(len(token) >= 6 for token in shared)


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
    normalized_title = normalize_text(title)
    normalized_text = normalize_text(text)
    if normalized_title in normalized_text:
        return True
    alias = canonical_entity_key(title)
    if len(alias) >= 5 and alias in canonical_entity_key(text):
        return True
    title_tokens = _entity_alias_tokens(title)
    text_tokens = _entity_alias_tokens(text)
    shared = title_tokens & text_tokens
    return len(shared) >= 2 or any(len(token) >= 6 for token in shared)
