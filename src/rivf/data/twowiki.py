import json
from pathlib import Path

from rivf.data.hotpotqa import parse_hotpot_style_record
from rivf.data.schema import QAItem


def load_2wikimultihopqa(path: str | Path) -> list[QAItem]:
    """2WikiMultiHopQA (Ho et al., COLING 2020) uses the same record shape as
    HotpotQA (context: [[title, [sentence, ...]], ...], supporting_facts:
    [[title, sent_id], ...]) -- verified field-for-field against a real
    record, not assumed from the README. Secondary benchmark, only added
    after the primary HotpotQA pipeline was validated, per the proposal."""
    raw = json.loads(Path(path).read_text())
    return [parse_hotpot_style_record(r) for r in raw]
