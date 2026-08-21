"""Reads results/*/results.jsonl and produces the paper's main comparison
table, the ablation table, and Experiment 3's trade-off (Pareto) curve.

Final pipeline state: title-prefixed embedding text + cross-encoder
reranking (BAAI/bge-reranker-v2-m3) for "Ours"'s semantic_score, replacing
the bi-encoder -- see version1.md sec. 10-11 for the full exploration
history. Operating point: alpha=0.6/budget=300 (moved down from 600 once
the reranker made the tight-budget regime the stronger, cleaner story).

Sources: main_comparison_v8_reranked (Ours, reranker, fresh LLM run),
main_comparison_v7_300tok (Dense RAG @ 300, fresh LLM run),
main_comparison_v6 (Fixed 1-hop/2-hop, unbounded, unaffected by any of
this so reused), reranker_search (Ours budget x alpha sweep with
reranker, no LLM), title_prefix_search (baseline budget sweep, bi-encoder
-- fixed_hop doesn't use the reranker), ablation_reranked_300
(random-pruning @ 300, reranker-independent).

Usage: python -m rivf.experiments.make_figures
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

BEST_ALPHA = 0.6
BEST_BUDGET = 300

METHOD_ORDER = ["ours", "dense_rag", "fixed_1hop", "fixed_2hop"]
METHOD_LABELS = {
    "ours": "Ours",
    "dense_rag": "Dense RAG",
    "fixed_1hop": "Fixed 1-hop",
    "fixed_2hop": "Fixed 2-hop",
}
# dataviz skill's validated categorical palette (light mode), assigned in a
# fixed narrative order -- Ours first, then baselines by sophistication --
# and held fixed rather than re-ordered per view.
METHOD_COLORS = {
    "ours": "#2a78d6",  # blue
    "dense_rag": "#eb6834",  # orange
    "fixed_1hop": "#1baf7a",  # aqua
    "fixed_2hop": "#eda100",  # yellow
}
METHOD_MARKERS = {"ours": "D", "dense_rag": "o", "fixed_1hop": "s", "fixed_2hop": "^"}

GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED_TEXT = "#898781"
PRIMARY_TEXT = "#0b0b0b"


def _load(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name / "results.jsonl"
    return pd.DataFrame(json.loads(line) for line in path.open())


def main_comparison_table() -> pd.DataFrame:
    ours = _load("main_comparison_v8_reranked")
    dense = _load("main_comparison_v7_300tok")
    dense = dense[dense["method"] == "dense_rag"]
    fixed = _load("main_comparison_v6")
    fixed = fixed[fixed["method"].isin(["fixed_1hop", "fixed_2hop"])]
    combined = pd.concat([ours, dense, fixed], ignore_index=True)
    cols = ["sp_recall", "sp_f1", "sp_prec", "em", "f1", "context_tokens", "num_selected"]
    return combined.groupby("method")[cols].mean().round(3).reindex(METHOD_ORDER)


def ablation_table() -> pd.DataFrame:
    sweep = _load("reranker_search")
    at_budget = sweep[(sweep["method"] == "ours_reranked") & (sweep["budget_tokens"] == BEST_BUDGET)]
    at_budget = at_budget.assign(method="ours")
    random_variant = _load("ablation_reranked_300")
    combined = pd.concat([at_budget, random_variant], ignore_index=True)
    cols = ["sp_recall", "sp_f1", "context_tokens"]
    return combined.groupby(["method", "alpha"], dropna=False)[cols].mean().round(3)


def controlled_comparison_table(dataset: str) -> pd.DataFrame:
    """Fair n=200 table where every CE variant uses the same reranker/budget."""
    experiment = {
        "hotpot": "fair_reranker_hotpot_200",
        "2wiki": "fair_reranker_2wiki_200",
    }[dataset]
    data = _load(experiment)
    cols = ["sp_recall", "sp_prec", "sp_f1", "em", "f1", "context_tokens", "num_selected"]
    return data.groupby("method")[cols].mean().round(3)


def controlled_alpha_table(dataset: str) -> pd.DataFrame:
    """Final-pipeline alpha ablation transferred unchanged across datasets."""
    experiment = {
        "hotpot": "fair_alpha_hotpot_200",
        "2wiki": "fair_alpha_2wiki_200",
    }[dataset]
    data = _load(experiment)
    cols = ["sp_recall", "sp_prec", "sp_f1", "context_tokens", "num_selected"]
    return data.groupby("alpha")[cols].mean().round(3)


def tradeoff_curve() -> Path:
    """Experiment 3: Supporting Fact Recall vs. context tokens, cropped to
    the evenly-swept 100-600 token range (the budget grid every method was
    actually tested at -- see Experiment 2). Baselines' natural unbounded
    operating points run to 800+ tokens and are deliberately left off this
    plot so the low-budget region the paper's claim is about isn't visually
    compressed by them; those numbers belong in the main comparison table
    instead. "Ours" comes from reranker_search (sliced to alpha=0.6);
    baselines (which don't use the reranker) come from title_prefix_search."""
    MAX_BUDGET = 600
    ours_sweep = _load("reranker_search")
    ours_sweep = ours_sweep[ours_sweep["alpha"] == BEST_ALPHA].assign(method="ours")
    baseline_sweep = _load("title_prefix_search")
    baseline_sweep = baseline_sweep[baseline_sweep["method"] != "ours"]
    sweep = pd.concat([ours_sweep, baseline_sweep], ignore_index=True)
    sweep = sweep[sweep["budget_tokens"] <= MAX_BUDGET]
    agg = sweep.groupby(["method", "budget_tokens"])[["context_tokens", "sp_recall"]].mean().reset_index()
    n_questions = ours_sweep["qid"].nunique()

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for method in METHOD_ORDER:
        sub = agg[agg["method"] == method].sort_values("context_tokens")
        if sub.empty:
            continue
        ax.plot(
            sub["context_tokens"],
            sub["sp_recall"],
            marker=METHOD_MARKERS[method],
            markersize=7,
            linewidth=2.5 if method == "ours" else 1.75,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            zorder=4 if method == "ours" else 3,
        )

    ax.set_xlim(80, 620)
    ax.set_xlabel("Context tokens (avg. per question)", color=PRIMARY_TEXT)
    ax.set_ylabel("Supporting Fact Recall", color=PRIMARY_TEXT)
    ax.set_title(
        f"Evidence recall vs. context cost\n({n_questions}-question HotpotQA dev-distractor subsample)",
        color=PRIMARY_TEXT,
    )
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=MUTED_TEXT)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "tradeoff_curve.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = tradeoff_curve()
    print(f"Wrote trade-off curve -> {path}\n")
    print("Main comparison table:")
    print(main_comparison_table().to_string())
    print("\nAblation table:")
    print(ablation_table().to_string())
    for dataset in ("hotpot", "2wiki"):
        print(f"\nControlled comparison ({dataset}, n=200):")
        print(controlled_comparison_table(dataset).to_string())
        print(f"\nControlled alpha ablation ({dataset}, n=200):")
        print(controlled_alpha_table(dataset).to_string())
