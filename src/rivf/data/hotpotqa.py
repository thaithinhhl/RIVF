import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from rivf.data.schema import Passage, QAItem, Sentence


def load_hotpotqa(
    path: str | Path,
    *,
    deduplicate_passages: bool = False,
    invalid_gold_policy: str = "keep",
) -> list[QAItem]:
    """Load Hotpot-style records with optional benchmark hygiene.

    Historical runs keep the raw records by default. Canonical GRAFT runs use
    ``deduplicate_passages=True`` so dense and graph methods see the same
    sentence identities, and ``invalid_gold_policy='exclude'`` so impossible
    supporting-fact annotations do not lower the metric ceiling.
    """
    if invalid_gold_policy not in {"keep", "exclude", "error"}:
        raise ValueError("invalid_gold_policy must be keep, exclude, or error")
    raw = json.loads(Path(path).read_text())
    items = []
    for original in raw:
        record = canonicalize_hotpot_style_record(original) if deduplicate_passages else original
        invalid = invalid_supporting_facts(record)
        if invalid:
            if invalid_gold_policy == "exclude":
                continue
            if invalid_gold_policy == "error":
                raise ValueError(
                    f"record {record.get('_id')} has invalid supporting facts: {invalid}"
                )
        items.append(parse_hotpot_style_record(record))
    return items


def canonicalize_hotpot_style_record(record: dict) -> dict:
    """Remove byte-identical duplicate passages without mutating raw data.

    Conflicting passages sharing a title are rejected because ``(title,
    sent_id)`` cannot uniquely identify both versions.
    """
    output = deepcopy(record)
    seen: dict[str, list[str]] = {}
    context = []
    for title, sentences in record["context"]:
        normalized = list(sentences)
        if title in seen:
            if seen[title] != normalized:
                raise ValueError(
                    f"record {record.get('_id')} has conflicting passages titled {title!r}"
                )
            continue
        seen[title] = normalized
        context.append([title, normalized])
    output["context"] = context
    return output


def invalid_supporting_facts(record: dict) -> list[tuple[str, int]]:
    passages: dict[str, list[list[str]]] = {}
    for title, sentences in record["context"]:
        passages.setdefault(title, []).append(sentences)
    return [
        (title, sent_id)
        for title, sent_id in record["supporting_facts"]
        if title not in passages
        or not isinstance(sent_id, int)
        or not any(0 <= sent_id < len(sentences) for sentences in passages[title])
    ]


def audit_hotpot_style_records(records: list[dict]) -> dict[str, int]:
    """Return deterministic integrity counts used by benchmark preflight."""
    duplicate_qids = len(records) - len({record["_id"] for record in records})
    duplicate_passage_records = 0
    conflicting_passage_records = 0
    invalid_gold_records = 0
    invalid_gold_facts = 0
    for record in records:
        title_counts = Counter(title for title, _ in record["context"])
        duplicates = {title for title, count in title_counts.items() if count > 1}
        duplicate_passage_records += bool(duplicates)
        for title in duplicates:
            versions = [sentences for current, sentences in record["context"] if current == title]
            if any(version != versions[0] for version in versions[1:]):
                conflicting_passage_records += 1
                break
        invalid = invalid_supporting_facts(record)
        invalid_gold_records += bool(invalid)
        invalid_gold_facts += len(invalid)
    return {
        "records": len(records),
        "duplicate_qids": duplicate_qids,
        "duplicate_passage_records": duplicate_passage_records,
        "conflicting_passage_records": conflicting_passage_records,
        "invalid_gold_records": invalid_gold_records,
        "invalid_gold_facts": invalid_gold_facts,
    }


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
