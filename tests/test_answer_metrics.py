from rivf.eval.answer_metrics import (
    answer_metrics,
    exact_match_score,
    f1_score,
    normalize_answer,
)


def test_exact_match():
    m = answer_metrics("Ed Wood", "Ed Wood")
    assert m["em"] == 1.0
    assert m["f1"] == 1.0
    assert m["prec"] == 1.0
    assert m["recall"] == 1.0


def test_partial_token_overlap():
    # prediction has one extra token -> perfect recall, imperfect precision
    f1, prec, recall = f1_score("the Ed Wood film", "Ed Wood")
    assert recall == 1.0
    assert round(prec, 4) == round(2 / 3, 4)
    assert round(f1, 4) == 0.8
    assert not exact_match_score("the Ed Wood film", "Ed Wood")


def test_no_overlap_is_zero():
    m = answer_metrics("Toronto", "Vancouver")
    assert m == {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}


def test_yes_no_matching_is_not_forced_zero():
    m = answer_metrics("yes", "yes")
    assert m["em"] == 1.0
    assert m["f1"] == 1.0


def test_yes_no_mismatch_is_forced_zero_not_partial_credit():
    # "no" vs "yes" share no tokens anyway, but this locks in the official
    # script's explicit special case rather than relying on incidental token overlap.
    m = answer_metrics("no", "yes")
    assert m == {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}


def test_yes_vs_non_yesno_gold_is_forced_zero():
    # normalized prediction is "yes" (a special token) but gold is a normal string
    # and they don't match -> official script forces ZERO_METRIC here too.
    m = answer_metrics("yes", "Ed Wood")
    assert m == {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}


def test_normalization_ignores_articles_case_and_trailing_punctuation():
    m = answer_metrics("The Ed Wood!", "ed wood")
    assert m["em"] == 1.0


def test_normalization_deletes_punctuation_without_inserting_space():
    # Matches the official script's actual (slightly quirky) behavior: remove_punc
    # deletes characters rather than replacing them with whitespace, so a hyphen
    # merges the two words it separated instead of keeping them apart.
    assert normalize_answer("Ed-Wood") == "edwood"
