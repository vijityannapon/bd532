"""Fairness audit (Section 3.2).

The model never sees `gender` or `age_group`; they are audited *after*
prediction because proxies (income, occupation, car ownership...) can still
carry group information.

Favourable outcome for an applicant = NOT flagged as HIGH_RISK.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def group_report(y_true, y_score, group, threshold: float) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score),
                       "g": np.asarray(group, dtype=object)}).dropna(subset=["g"])
    df["flag"] = (df["p"] >= threshold).astype(int)
    rows = []
    for g, d in df.groupby("g"):
        pos, neg = d[d.y == 1], d[d.y == 0]
        rows.append({
            "group": g,
            "n": len(d),
            "actual_default_rate": d.y.mean(),
            "mean_predicted_pd": d.p.mean(),
            "approval_rate": 1 - d.flag.mean(),
            "tpr_defaulters_caught": pos.flag.mean() if len(pos) else np.nan,
            "fpr_good_flagged": neg.flag.mean() if len(neg) else np.nan,
            "auc": roc_auc_score(d.y, d.p) if d.y.nunique() == 2 else np.nan,
        })
    return pd.DataFrame(rows).set_index("group")


def fairness_summary(report: pd.DataFrame) -> dict[str, float]:
    """Gap metrics across groups (0 = perfectly equal)."""
    ar = report["approval_rate"]
    return {
        "demographic_parity_diff": float(ar.max() - ar.min()),
        "disparate_impact_ratio": float(ar.min() / ar.max()) if ar.max() > 0 else float("nan"),
        "equal_opportunity_diff": float(report["tpr_defaulters_caught"].max()
                                        - report["tpr_defaulters_caught"].min()),
        "fpr_diff": float(report["fpr_good_flagged"].max() - report["fpr_good_flagged"].min()),
        "auc_diff": float(report["auc"].max() - report["auc"].min()),
    }


def audit(y_true, y_score, sensitive: pd.DataFrame, threshold: float) -> dict:
    """Run the audit for every sensitive attribute."""
    out = {}
    for attr in sensitive.columns:
        rep = group_report(y_true, y_score, sensitive[attr], threshold)
        out[attr] = {"by_group": rep, "summary": fairness_summary(rep)}
    return out
