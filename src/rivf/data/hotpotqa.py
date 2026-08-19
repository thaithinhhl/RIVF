import json
from pathlib import Path

from rivf.data.schema import Passage, QAItem, Sentence


def load_hotpotqa(path: str | Path) -> list[QAItem]:
    raw = json.loads(Path(path).read_text())
    return [parse_hotpot_style_record(r) for r in raw]


def parse_hotpot_style_record(record: dict) -> QAItem:
    """Shared by any dataset using HotpotQA's record shape -- 2WikiMultiHopQA
    is field-for-field identical for _id/question/answer/type/context/
    supporting_facts, differing only in having 4 type values instead of 2
    and no `level` field (defaults to "" below, same as here)."""
    passages = [
        Passage(
            title=title,
            sentences=[
                Sentence(passage_title=title, sent_id=i, text=text)
                for i, text in enumerate(sentences)
            ],
        )
        for title, sentences in record["context"]
    ]
    supporting_facts = [(title, sent_id) for title, sent_id in record["supporting_facts"]]
    return QAItem(
        qid=record["_id"],
        question=record["question"],
        answer=record.get("answer", ""),
        type=record.get("type", ""),
        level=record.get("level", ""),
        passages=passages,
        supporting_facts=supporting_facts,
    )
