"""Build RQ1--RQ6 tables from the 200-question v0.1 experiment outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rivf.data.hotpotqa import load_hotpotqa
from rivf.eval.efficiency import count_tokens


ROOT = Path("result_v0.1")
METRICS = (
    "sp_recall", "sp_prec", "sp_f1", "em", "f1", "context_tokens",
    "num_selected", "prompt_tokens", "output_tokens", "retrieval_latency_s",
    "llm_latency_s",
)
INPUT_USD_PER_M = 0.15
OUTPUT_USD_PER_M = 0.60


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dataset_name(row: dict) -> str:
    return "2Wiki" if "2wiki" in row["dataset"].lower() else "HotpotQA"


def mean(rows: list[dict], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(dataset_name(row), row["method"], row.get("budget_tokens"))].append(row)
    output = []
    for (dataset, method, budget), group in sorted(groups.items()):
        item = {"dataset": dataset, "method": method, "budget_tokens": budget, "n": len(group)}
        item.update({key: mean(group, key) for key in METRICS})
        if item["prompt_tokens"] is not None and item["output_tokens"] is not None:
            item["cost_usd_per_question"] = (
                item["prompt_tokens"] * INPUT_USD_PER_M
                + item["output_tokens"] * OUTPUT_USD_PER_M
            ) / 1_000_000
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def paired_ci(ours: list[dict], comparator: list[dict], seed: int = 0) -> tuple[float, float, float]:
    left = {r["qid"]: r["f1"] for r in ours}
    right = {r["qid"]: r["f1"] for r in comparator}
    qids = sorted(left.keys() & right.keys())
    delta = np.array([left[qid] - right[qid] for qid in qids], dtype=float)
    if not len(delta):
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    estimates = np.empty(10_000)
    for index in range(len(estimates)):
        estimates[index] = rng.choice(delta, len(delta), replace=True).mean()
    return float(delta.mean()), *map(float, np.percentile(estimates, [2.5, 97.5]))


def make_pareto(agg: list[dict], metric: str, filename: str, ylabel: str) -> None:
    rows = [r for r in agg if r.get(metric) is not None and r.get("budget_tokens") is not None]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, dataset in zip(axes, ("HotpotQA", "2Wiki")):
        for method in ("Dense CE", "Graph 1-hop CE", "Graph 2-hop CE", "Ours v2"):
            points = sorted(
                [r for r in rows if r["dataset"] == dataset and r["method"] == method],
                key=lambda r: r["context_tokens"],
            )
            if points:
                axis.plot([p["context_tokens"] for p in points], [p[metric] for p in points], marker="o", label=method)
        axis.set_title(dataset)
        axis.set_xlabel("Mean evidence tokens")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / filename, dpi=180)
    plt.close(fig)


def _score_only_selection(candidates: list[dict], score_key: str, budget: int = 200) -> set[tuple]:
    """Reproduce the score-only greedy cutoff from logged candidates."""
    ordered = sorted(candidates, key=lambda c: c.get(score_key) or 0.0, reverse=True)
    selected, used = set(), 0
    for candidate in ordered:
        cost = candidate["text_tokens"]
        if used + cost > budget:
            break
        selected.add((candidate["title"], candidate["sent_id"]))
        used += cost
    return selected


def rq5_analysis(main_rows: list[dict], ablation_rows: list[dict], diagnostic_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    main200 = {
        (dataset_name(r), r["qid"]): r for r in main_rows
        if r["method"] == "Ours v2" and r.get("budget_tokens") == 200
    }
    diag = {(dataset_name(r), r["qid"]): r for r in diagnostic_rows}
    no_graph_rows = {
        (dataset_name(r), r["qid"]): r for r in ablation_rows if r["method"] == "-GraphScore"
    }
    data_paths = {
        "HotpotQA": "data/raw/hotpot_dev_distractor_v1.json",
        "2Wiki": "data/raw/2wikimultihopqa_dev.json",
    }
    items = {dataset: {item.qid: item for item in load_hotpotqa(path)} for dataset, path in data_paths.items()}
    group_rows, oracle_rows, rescue_rows, failures = [], [], [], []
    for (dataset, qid), ours in main200.items():
        if (dataset, qid) not in diag or (dataset, qid) not in no_graph_rows:
            continue
        item = items[dataset][qid]
        gold = set(item.supporting_facts)
        ours_diag = diag[(dataset, qid)]
        candidates = ours_diag["candidates"]
        semantic = _score_only_selection(candidates, "semantic_score")
        graph = _score_only_selection(candidates, "graph_score")
        no_graph = no_graph_rows[(dataset, qid)]
        selected = {tuple(k) for k in ours["selected_keys"]}
        selected_no_graph = {tuple(k) for k in no_graph["selected_keys"]}
        candidate_hops = {(c["title"], c["sent_id"]): c.get("hop_distance") for c in candidates}
        candidate_keys = set(candidate_hops)
        if gold <= semantic:
            need_group = "Semantic-sufficient"
        elif gold <= (semantic | graph):
            need_group = "Semantic-blind"
        else:
            need_group = "Unrecoverable"
        group_rows.append({
            "dataset": dataset, "qid": qid, "question_type": item.type, "group": need_group,
            **{k: ours.get(k) for k in ("sp_recall", "sp_f1", "em", "f1", "context_tokens")},
        })
        gold_tokens = sum(count_tokens(s.text) for s in item.all_sentences() if s.key in gold)
        oracle_rows.append({
            "dataset": dataset, "qid": qid, "question_type": item.type,
            "one_hop_gold_recall": sum(k in candidate_hops and candidate_hops[k] <= 1 for k in gold) / len(gold),
            "two_hop_gold_recall": len(gold & candidate_keys) / len(gold),
            "full_gold_reachable": float(gold <= candidate_keys),
            "gold_tokens": gold_tokens, "gold_fit_200": float(gold_tokens <= 200),
        })
        added, displaced = selected - selected_no_graph, selected_no_graph - selected
        rescue_rows.append({
            "dataset": dataset, "qid": qid, "question_type": item.type,
            "activated": float(bool(added or displaced)),
            "correct_rescue": float(bool(added & gold)),
            "false_rescue": float(bool(added - gold)),
            "gold_added": len(added & gold), "gold_displaced": len(displaced & gold),
        })
        stages = []
        if not any(k in candidate_hops and candidate_hops[k] == 0 for k in gold): stages.append("Seed miss")
        if not gold <= candidate_keys: stages.append("Graph reachability miss")
        if gold_tokens > 200: stages.append("Budget infeasible")
        if added - gold: stages.append("Graph false rescue")
        if gold <= candidate_keys and gold_tokens <= 200 and not gold <= selected: stages.append("Pair/pruning error")
        if gold <= selected and ours.get("em") == 0: stages.append("Generation error")
        gold_conditional = [c for c in candidates if (c["title"], c["sent_id"]) in gold and c.get("conditional_score") is not None]
        if gold_conditional and not gold <= selected: stages.append("Conditional scoring error")
        expected_route = "direct_comparison" if item.type == "comparison" else ("bridge_comparison" if item.type == "bridge_comparison" else None)
        if expected_route and ours_diag.get("computed_route") != expected_route: stages.append("Router error")
        for stage in stages or ["No diagnosed failure"]:
            failures.append({"dataset": dataset, "qid": qid, "question_type": item.type, "stage": stage})
    return group_rows, oracle_rows, rescue_rows, failures


def grouped_means(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for group_key, members in sorted(groups.items()):
        result = dict(zip(keys, group_key)); result["n"] = len(members)
        result.update({metric: mean(members, metric) for metric in metrics})
        output.append(result)
    return output


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    main_rows = read_jsonl(ROOT / "rq1_rq2_rq3_hotpot_n200/results.jsonl") + read_jsonl(ROOT / "rq1_rq2_rq3_2wiki_n200/results.jsonl")
    hotpot_ablation = read_jsonl(ROOT / "rq4_ablation_hotpot_n200/results.jsonl")
    wiki_ablation = read_jsonl(ROOT / "rq4_ablation_2wiki_n200/results.jsonl")
    if not wiki_ablation:
        wiki_ablation = read_jsonl(ROOT / "rq4_ablation_2wiki_n200_retrieval/results.jsonl")
    ablation_rows = hotpot_ablation + wiki_ablation
    diagnostic_rows = read_jsonl(ROOT / "rq5_diagnostic_hotpot_n200/results.jsonl") + read_jsonl(ROOT / "rq5_diagnostic_2wiki_n200/results.jsonl")
    agg = aggregate(main_rows + ablation_rows)
    write_csv(ROOT / "aggregate_metrics.csv", agg)
    make_pareto(agg, "f1", "rq3_answer_f1_pareto.png", "Answer F1")
    make_pareto(agg, "sp_f1", "rq3_sp_f1_pareto.png", "SP F1")

    ci_rows = []
    for dataset in ("HotpotQA", "2Wiki"):
        ours = [r for r in main_rows if dataset_name(r) == dataset and r["method"] == "Ours v2" and r.get("budget_tokens") == 200]
        for method in ("Dense CE", "Graph 1-hop CE", "Graph 2-hop CE"):
            comp = [r for r in main_rows if dataset_name(r) == dataset and r["method"] == method and r.get("budget_tokens") == 600]
            delta, low, high = paired_ci(ours, comp)
            ci_rows.append({"dataset": dataset, "comparator": method, "delta_answer_f1": delta, "ci_low": low, "ci_high": high, "non_inferior_margin_-0.02": low >= -0.02})
    write_csv(ROOT / "rq1_paired_bootstrap_ci.csv", ci_rows)

    group_rows, oracle_rows, rescue_rows, failures = rq5_analysis(main_rows, ablation_rows, diagnostic_rows) if main_rows and ablation_rows and diagnostic_rows else ([], [], [], [])
    type_summary = grouped_means(group_rows, ("dataset", "question_type"), ("sp_recall", "sp_f1", "em", "f1", "context_tokens"))
    need_summary = grouped_means(group_rows, ("dataset", "group"), ("sp_recall", "sp_f1", "em", "f1", "context_tokens"))
    oracle_summary = grouped_means(oracle_rows, ("dataset",), ("one_hop_gold_recall", "two_hop_gold_recall", "full_gold_reachable", "gold_tokens", "gold_fit_200"))
    rescue_summary = grouped_means(rescue_rows, ("dataset",), ("activated", "correct_rescue", "false_rescue", "gold_added", "gold_displaced"))
    failure_summary = [{"dataset": d, "stage": s, "n": n} for (d, s), n in sorted(Counter((r["dataset"], r["stage"]) for r in failures).items())]
    for filename, rows in (("rq5_type.csv", type_summary), ("rq5_graph_need.csv", need_summary), ("rq5_oracle.csv", oracle_summary), ("rq5_rescue.csv", rescue_summary), ("rq5_failures.csv", failure_summary)):
        write_csv(ROOT / filename, rows)

    completed = defaultdict(set)
    for row in main_rows:
        completed[dataset_name(row)].add(row["qid"])
    by_key = {(r["dataset"], r["method"], r["budget_tokens"]): r for r in agg}
    def result_row(dataset: str, method: str, budget: int) -> dict:
        return by_key.get((dataset, method, budget), {})

    lines = ["# Result v0.1 — target 200 questions per benchmark", "", "All values are pilot/subsample results, not full-benchmark claims. `N < 200` means the OpenAI daily quota interrupted generation; retrieval-only rows deliberately leave Answer metrics blank.", "", "## Run status", "", "| Dataset | Complete paired QIDs | Target | Main status | RQ4 status |", "|---|---:|---:|---|---|"]
    for dataset in ("HotpotQA", "2Wiki"):
        n_complete = min((len({r['qid'] for r in main_rows if dataset_name(r) == dataset and r['method'] == method and r.get('budget_tokens') == 200}) for method in ("Dense CE", "Graph 1-hop CE", "Graph 2-hop CE", "Ours v2")), default=0)
        rq4_answer = all(result_row(dataset, method, 200).get("f1") is not None for method in ("-GraphScore", "Uniform PPR", "-DisagreementGate", "-Router", "-ConditionalScoring", "-PairPreservation", "Graph-only"))
        lines.append(f"| {dataset} | {n_complete} | 200 | {'Complete' if n_complete == 200 else 'Partial — API quota'} | {'Complete' if rq4_answer else 'Retrieval complete; Answer pending'} |")

    lines += ["", "## RQ1 — Main comparison: baseline 600 vs Ours 200", "", "| Dataset | Method | Budget | N | SP Recall | SP Precision | SP F1 | Answer EM | Answer F1 | Evidence tokens |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("HotpotQA", "2Wiki"):
        for method, budget in (("Dense CE", 600), ("Graph 1-hop CE", 600), ("Graph 2-hop CE", 600), ("Ours v2", 200)):
            row = result_row(dataset, method, budget)
            lines.append(f"| {dataset} | {method} | {budget} | {row.get('n','—')} | {fmt(row.get('sp_recall'))} | {fmt(row.get('sp_prec'))} | {fmt(row.get('sp_f1'))} | {fmt(row.get('em'))} | {fmt(row.get('f1'))} | {fmt(row.get('context_tokens'),1)} |")

    lines += ["", "### RQ1 paired non-inferiority", "", "| Dataset | Comparator 600 | Δ Answer F1 | 95% CI | Non-inferior at −0.02 |", "|---|---|---:|---|---|"]
    for row in ci_rows:
        lines.append(f"| {row['dataset']} | {row['comparator']} | {fmt(row['delta_answer_f1'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {row['non_inferior_margin_-0.02']} |")

    lines += ["", "## RQ2 — Same-budget control at 200 tokens", "", "| Dataset | Method | N | SP Recall | SP Precision | SP F1 | Answer EM | Answer F1 | Evidence tokens |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("HotpotQA", "2Wiki"):
        for method in ("Dense CE", "Graph 1-hop CE", "Graph 2-hop CE", "Ours v2"):
            row = result_row(dataset, method, 200)
            lines.append(f"| {dataset} | {method} | {row.get('n','—')} | {fmt(row.get('sp_recall'))} | {fmt(row.get('sp_prec'))} | {fmt(row.get('sp_f1'))} | {fmt(row.get('em'))} | {fmt(row.get('f1'))} | {fmt(row.get('context_tokens'),1)} |")

    for metric, title in (("f1", "Answer F1"), ("sp_f1", "SP F1")):
        lines += ["", f"## RQ3 — {title} theo token budget", "", "| Dataset | Method | 100 | 150 | 200 | 300 | 400 | 600 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for dataset in ("HotpotQA", "2Wiki"):
            for method in ("Dense CE", "Graph 1-hop CE", "Graph 2-hop CE", "Ours v2"):
                values = [fmt(result_row(dataset, method, budget).get(metric)) for budget in (100, 150, 200, 300, 400, 600)]
                lines.append(f"| {dataset} | {method} | " + " | ".join(values) + " |")
    lines += ["", "Figures: [`rq3_answer_f1_pareto.png`](rq3_answer_f1_pareto.png) and [`rq3_sp_f1_pareto.png`](rq3_sp_f1_pareto.png)."]

    lines += ["", "## RQ4 — Component ablation at 200 tokens", "", "| Dataset | Variant | N | SP Recall | SP Precision | SP F1 | Answer EM | Answer F1 | Tokens | Status |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    variants = ("Ours v2", "-GraphScore", "Uniform PPR", "-DisagreementGate", "-Router", "-ConditionalScoring", "-PairPreservation", "Graph-only")
    for dataset in ("HotpotQA", "2Wiki"):
        for variant in variants:
            row = result_row(dataset, variant, 200)
            if row.get("f1") is None:
                status = "Retrieval-only; Answer pending"
            elif row.get("n") != 200:
                status = "Partial — API quota"
            else:
                status = "Complete"
            lines.append(f"| {dataset} | {variant} | {row.get('n','—')} | {fmt(row.get('sp_recall'))} | {fmt(row.get('sp_prec'))} | {fmt(row.get('sp_f1'))} | {fmt(row.get('em'))} | {fmt(row.get('f1'))} | {fmt(row.get('context_tokens'),1)} | {status} |")

    lines += ["", "## RQ5 — Robustness and error analysis", "", "### Breakdown theo question type", "", "| Dataset | Type | N | SP Recall | SP F1 | Answer EM | Answer F1 | Tokens |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in type_summary:
        lines.append(f"| {row['dataset']} | {row['question_type']} | {row['n']} | {fmt(row['sp_recall'])} | {fmt(row['sp_f1'])} | {fmt(row['em'])} | {fmt(row['f1'])} | {fmt(row['context_tokens'],1)} |")
    lines += ["", "### Mức độ cần graph", "", "| Dataset | Group | N | SP Recall | SP F1 | Answer F1 | Tokens |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in need_summary:
        lines.append(f"| {row['dataset']} | {row['group']} | {row['n']} | {fmt(row['sp_recall'])} | {fmt(row['sp_f1'])} | {fmt(row['f1'])} | {fmt(row['context_tokens'],1)} |")
    lines += ["", "### Oracle reachability", "", "| Dataset | N | 1-hop gold recall | 2-hop gold recall | Full gold reachable | Mean gold tokens | Gold fit ≤200 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in oracle_summary:
        lines.append(f"| {row['dataset']} | {row['n']} | {fmt(row['one_hop_gold_recall'])} | {fmt(row['two_hop_gold_recall'])} | {fmt(row['full_gold_reachable'])} | {fmt(row['gold_tokens'],1)} | {fmt(row['gold_fit_200'])} |")
    lines += ["", "### Graph rescue diagnostic", "", "| Dataset | N | Activated | Correct rescue | False rescue | Mean gold added | Mean gold displaced |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rescue_summary:
        lines.append(f"| {row['dataset']} | {row['n']} | {fmt(row['activated'])} | {fmt(row['correct_rescue'])} | {fmt(row['false_rescue'])} | {fmt(row['gold_added'])} | {fmt(row['gold_displaced'])} |")
    lines += ["", "Failure counts are multi-label and stored in [`rq5_failures.csv`](rq5_failures.csv)."]

    lines += ["", "## RQ6 — Efficiency and cost", "", "| Dataset | Method | Budget | N | Evidence tok | Prompt tok | Output tok | Nodes | Retrieval s | LLM s | USD/question |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("HotpotQA", "2Wiki"):
        for method, budget in (("Dense CE", 600), ("Graph 1-hop CE", 600), ("Graph 2-hop CE", 600), ("Ours v2", 200)):
            row = result_row(dataset, method, budget)
            lines.append(f"| {dataset} | {method} | {budget} | {row.get('n','—')} | {fmt(row.get('context_tokens'),1)} | {fmt(row.get('prompt_tokens'),1)} | {fmt(row.get('output_tokens'),1)} | {fmt(row.get('num_selected'),1)} | {fmt(row.get('retrieval_latency_s'),3)} | {fmt(row.get('llm_latency_s'),3)} | {fmt(row.get('cost_usd_per_question'),6)} |")
    lines += ["", "RQ6 latency hiện là instrumented pilot trong shared run. Trước khi dùng làm final paper claim, cần đo lại từng method bằng process riêng như protocol trong `experiment_v2.md`."]

    lines += ["", "## Reproducibility artifacts", "", "- Full aggregate: [`aggregate_metrics.csv`](aggregate_metrics.csv)", "- Paired bootstrap: [`rq1_paired_bootstrap_ci.csv`](rq1_paired_bootstrap_ci.csv)", "- RQ5 detail: [`rq5_type.csv`](rq5_type.csv), [`rq5_graph_need.csv`](rq5_graph_need.csv), [`rq5_oracle.csv`](rq5_oracle.csv), [`rq5_rescue.csv`](rq5_rescue.csv), [`rq5_failures.csv`](rq5_failures.csv)", "", "## Accounting", "", f"Cost uses gpt-4o-mini standard rates: ${INPUT_USD_PER_M}/1M input and ${OUTPUT_USD_PER_M}/1M output tokens. Evidence, prompt, and output tokens are reported separately."]
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote summaries -> {ROOT}")


if __name__ == "__main__":
    main()
