from pathlib import Path

from rivf.data.hotpotqa import load_hotpotqa
from rivf.data.schema import QAItem


def load_2wikimultihopqa(
    path: str | Path,
    *,
    deduplicate_passages: bool = False,
    invalid_gold_policy: str = "keep",
) -> list[QAItem]:
    """2WikiMultiHopQA (Ho et al., COLING 2020) uses the same record shape as
    HotpotQA (context: [[title, [sentence, ...]], ...], supporting_facts:
    [[title, sent_id], ...]) -- verified field-for-field against a real
    record, not assumed from the README. Secondary benchmark, only added
    after the primary HotpotQA pipeline was validated, per the proposal."""
    return load_hotpotqa(
        path,
        deduplicate_passages=deduplicate_passages,
        invalid_gold_policy=invalid_gold_policy,
    )
