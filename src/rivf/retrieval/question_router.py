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
