import pandas as pd

from rivf.experiments.paired_stats import paired_bootstrap


def test_paired_bootstrap_uses_within_question_differences():
    reference = pd.Series([0.8, 0.6, 1.0], index=["q1", "q2", "q3"])
    comparator = pd.Series([0.5, 0.4, 0.7], index=["q1", "q2", "q3"])
    mean, low, high = paired_bootstrap(reference, comparator, n_resamples=2_000, seed=0)
    assert round(mean, 4) == round((0.3 + 0.2 + 0.3) / 3, 4)
    assert low > 0
    assert high > 0


def test_paired_bootstrap_aligns_rows_by_question_id():
    reference = pd.Series([0.9, 0.2], index=["q1", "q2"])
    comparator = pd.Series([0.1, 0.8], index=["q2", "q1"])
    mean, _, _ = paired_bootstrap(reference, comparator, n_resamples=500, seed=0)
    assert round(mean, 4) == 0.1
