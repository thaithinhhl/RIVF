from rivf.eval.supporting_facts import supporting_fact_metrics


def test_perfect_match():
    gold = [("A", 0), ("B", 1)]
    m = supporting_fact_metrics(gold, gold)
    assert m == {"sp_em": 1.0, "sp_f1": 1.0, "sp_prec": 1.0, "sp_recall": 1.0}


def test_partial_overlap():
    gold = [("A", 0), ("B", 1)]
    predicted = [("A", 0), ("C", 2)]
    m = supporting_fact_metrics(predicted, gold)
    assert m["sp_prec"] == 0.5
    assert m["sp_recall"] == 0.5
    assert m["sp_f1"] == 0.5
    assert m["sp_em"] == 0.0


def test_empty_prediction():
    gold = [("A", 0)]
    m = supporting_fact_metrics([], gold)
    assert m == {"sp_em": 0.0, "sp_f1": 0.0, "sp_prec": 0.0, "sp_recall": 0.0}


def test_extra_predictions_hurt_precision_not_recall():
    gold = [("A", 0)]
    predicted = [("A", 0), ("B", 1), ("C", 2)]
    m = supporting_fact_metrics(predicted, gold)
    assert m["sp_recall"] == 1.0
    assert round(m["sp_prec"], 4) == round(1 / 3, 4)
    assert m["sp_em"] == 0.0  # extra false positives break exact match


def test_accepts_lists_like_raw_json_not_just_tuples():
    gold = [["A", 0]]
    predicted = [["A", 0]]
    m = supporting_fact_metrics(predicted, gold)
    assert m["sp_em"] == 1.0
