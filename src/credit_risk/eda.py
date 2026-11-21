"""Basic input data exploration (Section 3.1).

Usage:  python -m credit_risk.eda [--data PATH]
Writes reports/eda/eda_summary.md and PNG charts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config, data, features


def run_eda(raw: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    apps = features.raw_to_applicants(raw)
    X, y, sens = features.prepare_dataset(raw)
    df = X.join(y).join(sens)
    lines = ["# Exploratory Data Analysis", ""]
    lines += [f"* Source: {'SYNTHETIC sample' if data.is_synthetic(raw) else 'Home Credit application_train.csv'}",
              f"* Rows: {len(raw):,}  |  raw columns used: {raw.shape[1]}  |  model features: {X.shape[1]}",
              f"* Default rate (TARGET=1): **{y.mean():.2%}**  ->  class imbalance "
              f"{(1 - y.mean()) / y.mean():.1f} : 1", ""]

    lines += ["## 1. Data types", "", "| column | dtype | missing % | unique |", "|---|---|---|---|"]
    for c in raw.columns:
        lines.append(f"| {c} | {raw[c].dtype} | {raw[c].isna().mean():.1%} | {raw[c].nunique():,} |")

    lines += ["", "## 2. Numeric statistics (after unit conversion)", "",
              X.select_dtypes("number").describe().T.round(3).to_markdown(), ""]

    lines += ["## 3. Data-quality findings", "",
              f"* `DAYS_EMPLOYED == 365243` sentinel (pensioners/unemployed): "
              f"{(raw['DAYS_EMPLOYED'] == config.DAYS_EMPLOYED_SENTINEL).mean():.1%} of rows -> set to NaN + flag `employment_missing`.",
              f"* `EXT_SOURCE_1` missing {raw['EXT_SOURCE_1'].isna().mean():.1%}, "
              f"`OCCUPATION_TYPE` missing {raw['OCCUPATION_TYPE'].isna().mean():.1%} -> imputation + missing indicators.",
              f"* `CODE_GENDER` values: {raw['CODE_GENDER'].value_counts().to_dict()} ('XNA' -> unknown).", ""]

    lines += ["## 4. Default rate by category", ""]
    for c in ["education", "income_type", "contract_type", "housing_type"]:
        t = df.groupby(c)[config.TARGET].agg(["count", "mean"]).rename(columns={"mean": "default_rate"})
        lines += [f"### {c}", "", t.sort_values("default_rate").round(4).to_markdown(), ""]
    for c in config.SENSITIVE_ATTRIBUTES:
        t = df.groupby(c)[config.TARGET].agg(["count", "mean"]).rename(columns={"mean": "default_rate"})
        lines += [f"### {c} (sensitive - audit only)", "", t.round(4).to_markdown(), ""]

    corr = df.select_dtypes("number").corr()[config.TARGET].drop(config.TARGET).sort_values()
    lines += ["## 5. Correlation of numeric features with default", "",
              corr.round(4).to_frame("pearson_r").to_markdown(), ""]

    # ---- charts -----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    y.value_counts().sort_index().plot.bar(ax=ax[0], color=["#4C78A8", "#E45756"])
    ax[0].set_title("Class distribution (0 = repaid, 1 = default)")
    raw.isna().mean().sort_values(ascending=False).head(8).plot.barh(ax=ax[1], color="#72B7B2")
    ax[1].set_title("Missing rate (top 8)")
    for t, col in ((0, "#4C78A8"), (1, "#E45756")):
        apps.loc[apps[config.RAW_TARGET] == t, "age_years"].plot.hist(
            ax=ax[2], bins=30, alpha=0.55, density=True, color=col, label=f"target={t}")
    ax[2].set_title("Age distribution by target"); ax[2].legend()
    fig.tight_layout(); fig.savefig(out_dir / "eda_overview.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    corr.plot.barh(ax=ax, color=["#E45756" if v > 0 else "#4C78A8" for v in corr])
    ax.set_title("Pearson correlation with default"); fig.tight_layout()
    fig.savefig(out_dir / "eda_correlation.png", dpi=120); plt.close(fig)

    lines += ["![overview](eda_overview.png)", "", "![correlation](eda_correlation.png)", ""]
    path = out_dir / "eda_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":  # pragma: no cover
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--nrows", type=int, default=None)
    a = p.parse_args()
    print("EDA written to", run_eda(data.load_raw(a.data, a.nrows), config.REPORTS_DIR / "eda"))
