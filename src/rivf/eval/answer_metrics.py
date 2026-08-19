import re
import string
from collections import Counter

ZERO_METRIC = (0.0, 0.0, 0.0)


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Ported verbatim from the official hotpot_evaluate_v1.py f1_score.

    Yes/no/noanswer predictions or golds that don't match exactly are forced to
    (0, 0, 0) rather than falling through to token overlap — otherwise a
    comparison-question answer of "no" vs gold "yes" would still earn partial
    credit for the "n" token overlap, which the official script deliberately avoids.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction in ("yes", "no", "noanswer") and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ("yes", "no", "noanswer") and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def answer_metrics(prediction: str, gold: str) -> dict[str, float]:
    """Per-question answer EM/F1/precision/recall.

    Returns one row; macro-average across questions by taking the column-wise
    mean over many of these rows (e.g. pandas .mean()), matching how the
    official script sums per-question metrics then divides by N.
    """
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    return {"em": float(em), "f1": f1, "prec": prec, "recall": recall}
