"""Fairness comparison chart for the report (Section 3.2).

Reads the per-version fairness CSVs written by `credit_risk.train` (validation set)
and writes reports/figures/fairness_comparison.png.
Usage:  python scripts/plot_fairness.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG = Path(__file__).resolve().parents[1] / "reports" / "figures"
VERSIONS = {"v5_hgb_tuned": "v5 (no mitigation)", "v6_hgb_tuned_reweighed": "v6 (reweighing)"}
METRICS = [("approval_rate", "Approval rate"), ("fpr_good_flagged", "Good customers flagged (FPR)"),
           ("tpr_defaulters_caught", "Defaulters caught (TPR)")]
COLORS = {"F": "#4C78A8", "M": "#F58518", "<30": "#E45756", "30-50": "#72B7B2", ">50": "#54A24B"}


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, attr in enumerate(["gender", "age_group"]):
        frames = {label: pd.read_csv(FIG / f"{v}_fairness_{attr}.csv", index_col="group")
                  for v, label in VERSIONS.items()}
        groups = list(next(iter(frames.values())).index)
        if attr == "age_group":
            groups = [g for g in ["<30", "30-50", ">50"] if g in groups]
        for col, (metric, title) in enumerate(METRICS):
            ax = axes[row, col]
            width = 0.8 / len(groups)
            for i, g in enumerate(groups):
                vals = [frames[label].loc[g, metric] for label in VERSIONS.values()]
                xs = [x + i * width for x in range(len(VERSIONS))]
                bars = ax.bar(xs, vals, width, label=str(g), color=COLORS.get(str(g)))
                ax.bar_label(bars, fmt="%.2f", fontsize=8)
            ax.set_xticks([x + width * (len(groups) - 1) / 2 for x in range(len(VERSIONS))])
            ax.set_xticklabels(VERSIONS.values(), fontsize=9)
            ax.set_title(f"{title} - {attr}", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.25)  # headroom for the legend
            if col == 0:
                ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=len(groups))
    fig.suptitle("Fairness by group, validation set (threshold = cost-optimal)", fontsize=12)
    fig.tight_layout()
    out = FIG / "fairness_comparison.png"
    fig.savefig(out, dpi=140)
    print("Wrote", out)


if __name__ == "__main__":
    main()
