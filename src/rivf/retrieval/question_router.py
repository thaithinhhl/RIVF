"""Dataset-agnostic routing from question text to an evidence policy.

The router deliberately does not inspect ``QAItem.type``: benchmark labels
are evaluation metadata and are unavailable for a real user question.
"""

import re

DIRECT_COMPARISON = "direct_comparison"
BRIDGE_COMPARISON = "bridge_comparison"
CHAINED_REASONING = "chained_reasoning"

_COMPARISON_RE = re.compile(
    r"\b(?:same|both|earlier|later|older|younger|first|more recently|"
    r"higher|lower|larger|smaller|more|less|share)\b",
    re.IGNORECASE,
)

# A comparison involving one of these relations still needs an intermediate
# entity before the final comparison (e.g. compare two films' directors).
_BRIDGE_RELATION_RE = re.compile(
    r"\b(?:director|author|writer|creator|producer|founder|owner|leader|"
    r"father|mother|parent|grandfather|grandmother|son|daughter|child|"
    r"wife|husband|spouse|brother|sister|coach|manager|president)\b",
    re.IGNORECASE,
)

# V2 keeps the frozen v1 router intact but recognizes common plural and
# irregular relation nouns. For example, "do both films have directors from
# the same country?" is a bridge comparison even though v1's singular-only
# cue routes it as direct comparison.
_BRIDGE_RELATION_V2_RE = re.compile(
    r"\b(?:directors?|authors?|writers?|creators?|producers?|founders?|owners?|"
    r"leaders?|fathers?|mothers?|parents?|grandfathers?|grandmothers?|sons?|"
    r"daughters?|children|child|wives|wife|husbands?|spouses?|brothers?|"
    r"sisters?|coaches|coach|managers?|presidents?)\b",
    re.IGNORECASE,
)


def route_question(question: str) -> str:
    """Choose semantic comparison or chained conditional evidence retrieval.

    Direct comparisons usually name both target entities in the question and
    are already handled well by semantic CE ranking. Everything else is sent
    through conditional pair reranking, including compositional, inference,
    bridge, and bridge-comparison questions.
    """
    is_comparison = bool(_COMPARISON_RE.search(question))
    has_intermediate_relation = bool(_BRIDGE_RELATION_RE.search(question))
    if is_comparison:
        return BRIDGE_COMPARISON if has_intermediate_relation else DIRECT_COMPARISON
    return CHAINED_REASONING


def route_question_v2(question: str) -> str:
    """Plural-aware router used only by GRAFT-v2.

    Keeping a separate entry point protects frozen GRAFT-v1 reproduction.
    """
    is_comparison = bool(_COMPARISON_RE.search(question))
    has_intermediate_relation = bool(_BRIDGE_RELATION_V2_RE.search(question))
    if is_comparison:
        return BRIDGE_COMPARISON if has_intermediate_relation else DIRECT_COMPARISON
    return CHAINED_REASONING
