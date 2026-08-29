import json

import pytest

from rivf.data.hotpotqa import (
    audit_hotpot_style_records,
    load_hotpotqa,
)


def _record():
    return {
        "_id": "q1",
        "question": "q",
        "answer": "a",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": [["A", 0]],
        "context": [["A", ["one"]], ["A", ["one"]], ["B", ["two"]]],
    }


def test_loader_deduplicates_identical_passages(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([_record()]))
    item = load_hotpotqa(path, deduplicate_passages=True)[0]
    assert [passage.title for passage in item.passages] == ["A", "B"]


def test_loader_excludes_records_with_invalid_gold(tmp_path):
    record = _record()
    record["supporting_facts"] = [["A", 99]]
    path = tmp_path / "data.json"
    path.write_text(json.dumps([record]))
    assert load_hotpotqa(path, invalid_gold_policy="exclude") == []


def test_loader_rejects_conflicting_duplicate_titles(tmp_path):
    record = _record()
    record["context"][1] = ["A", ["different"]]
    path = tmp_path / "data.json"
    path.write_text(json.dumps([record]))
    with pytest.raises(ValueError, match="conflicting passages"):
        load_hotpotqa(path, deduplicate_passages=True)


def test_audit_reports_duplicate_and_invalid_records():
    record = _record()
    record["supporting_facts"] = [["A", 99]]
    audit = audit_hotpot_style_records([record])
    assert audit["duplicate_passage_records"] == 1
    assert audit["invalid_gold_records"] == 1
