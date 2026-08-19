from rivf.methods.base import Candidate
from rivf.retrieval.pruning import prune

# Calibrated on a 200-question HotpotQA dev holdout (version1.md sec. 6c):
# the 80th percentile of need_graph on that sample. Above it, semantic-only
# and graph-only selections disagree enough / semantic confidence is low
# enough that the question resembles the "semantic-blind" bucket found by
# the gold-requiring diagnostic split in sec. 6b -- this estimator uses only
# signals available at inference time (no gold) to approximate that label.
NEED_GRAPH_THRESHOLD = 0.36
LOW_ALPHA = 0.2


def _need_graph_score(
    candidates: list[Candidate], budget_tokens: int | None, top_k: int | None
) -> float:
    """Three signals computable without gold evidence, combined into one
    "does this question need graph help" score:

    - Jaccard disagreement between a semantic-only and a graph-only top-K
      selection under the same operating point (low overlap => the two
      rankings point at different evidence -- graph may be catching
      something semantic missed).
    - Absolute semantic confidence: the highest semantic_score in the whole
      candidate pool (low even at the top => the scorer itself isn't
      confident anything here is relevant).
    - seed_miss_frac: fraction of seed-level (hop=0) candidates the
      semantic-only selection didn't keep (graph "knows" about a seed that
      ranked low semantically).

    Validated on a 200-question calibration set: AUC=0.741 for flagging
    questions in the true semantic-blind/unrecoverable buckets (version1.md
    sec. 6b/6c) -- a real but partial signal, not a solved classifier."""
    if not candidates:
        return 0.0

    s_sem = prune(candidates, alpha=1.0, strategy="topk", budget_tokens=budget_tokens, top_k=top_k)
    s_graph = prune(candidates, alpha=0.0, strategy="topk", budget_tokens=budget_tokens, top_k=top_k)
    sem_keys = {s.key for s in s_sem}
    graph_keys = {s.key for s in s_graph}
    union = sem_keys | graph_keys
    jaccard = len(sem_keys & graph_keys) / len(union) if union else 1.0

    max_sem = max((c.semantic_score or 0.0 for c in candidates), default=0.0)

    seeds = [c for c in candidates if c.hop_distance == 0]
    seed_miss_frac = (
        sum(1 for c in seeds if c.sentence.key not in sem_keys) / len(seeds) if seeds else 0.0
    )

    return (1 - jaccard) * 0.5 + (1 - max_sem) * 0.3 + seed_miss_frac * 0.2


def estimate_alpha(
    candidates: list[Candidate],
    base_alpha: float = 0.6,
    budget_tokens: int | None = 300,
    top_k: int | None = None,
) -> float:
    """Proxy Confidence Estimator: picks alpha per question instead of using
    one fixed value for the whole dataset (see _need_graph_score for the
    three signals it's built from). A coarse two-value switch (base_alpha
    vs. LOW_ALPHA) at one calibrated threshold, not a continuously tuned
    function -- see estimate_alpha_continuous for that variant. Pass
    exactly one of budget_tokens/top_k, matching whichever operating point
    the rest of the pipeline is using (token budget vs. fixed node count)."""
    if not candidates:
        return base_alpha
    need_graph = _need_graph_score(candidates, budget_tokens, top_k)
    return LOW_ALPHA if need_graph >= NEED_GRAPH_THRESHOLD else base_alpha


def estimate_alpha_continuous(
    candidates: list[Candidate],
    base_alpha: float = 0.6,
    budget_tokens: int | None = 300,
    top_k: int | None = None,
    min_alpha: float = 0.1,
    max_alpha: float = 0.9,
) -> float:
    """Continuous version of estimate_alpha: shifts base_alpha down by
    need_graph directly and clamps to [min_alpha, max_alpha], instead of
    switching between two fixed values at one threshold. A high-disagreement
    question (need_graph close to its empirical max, ~0.7) lands alpha near
    min_alpha; a low-disagreement one (need_graph near 0) stays near
    base_alpha."""
    if not candidates:
        return base_alpha
    need_graph = _need_graph_score(candidates, budget_tokens, top_k)
    return max(min_alpha, min(max_alpha, base_alpha - need_graph))
