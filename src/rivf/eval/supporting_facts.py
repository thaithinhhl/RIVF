def supporting_fact_metrics(
    predicted: list[tuple[str, int]], gold: list[tuple[str, int]]
) -> dict[str, float]:
    """Per-question supporting-fact P/R/F1/EM, ported verbatim from
    hotpot_evaluate_v1.py's update_sp (set comparison of (title, sent_id) pairs).

    Returns one row; macro-average across questions by taking the column-wise
    mean over many of these rows, matching the official script's sum-then-divide.
    """
    cur_sp_pred = set(map(tuple, predicted))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return {"sp_em": em, "sp_f1": f1, "sp_prec": prec, "sp_recall": recall}
