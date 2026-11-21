"""Evaluation metrics, cost-based threshold selection and decision bands."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: max separation between good/bad score CDFs (credit-scoring standard)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def expected_cost(y_true, y_score, threshold: float,
                  c_fn: float = config.COST_FALSE_NEGATIVE,
                  c_fp: float = config.COST_FALSE_POSITIVE) -> float:
    """Average misclassification cost per application."""
    y_true = np.asarray(y_true)
    pred = np.asarray(y_score) >= threshold
    fn = np.sum((~pred) & (y_true == 1))
    fp = np.sum(pred & (y_true == 0))
    return float((c_fn * fn + c_fp * fp) / len(y_true))


def select_threshold(y_true, y_score, c_fn: float = config.COST_FALSE_NEGATIVE,
                     c_fp: float = config.COST_FALSE_POSITIVE) -> float:
    """Pick the probability threshold minimising expected business cost."""
    grid = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 197)))
    costs = [expected_cost(y_true, y_score, t, c_fn, c_fp) for t in grid]
    return float(grid[int(np.argmin(costs))])


def classification_metrics(y_true, y_score, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "flag_rate": float(pred.mean()),
        "expected_cost": expected_cost(y_true, y_score, threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def decision_band(pd_score: float, decline_threshold: float,
                  review_factor: float = config.REVIEW_BAND_FACTOR) -> str:
    """Map a probability of default to a *recommendation* for the loan officer.

    The AI never auto-declines: high-risk cases are flagged for a human decision.
    """
    if pd_score >= decline_threshold:
        return "HIGH_RISK"
    if pd_score >= decline_threshold * review_factor:
        return "REVIEW"
    return "LOW_RISK"


def risk_score(pd_score: float) -> int:
    """0-1000 score (higher = safer), a familiar format for credit officers.

    Points-to-double-odds scaling: 600 points at odds 50:1 good:bad, PDO = 40.
    """
    p = float(np.clip(pd_score, 1e-6, 1 - 1e-6))
    odds_good = (1 - p) / p
    score = 600 + 40 / np.log(2) * np.log(odds_good / 50)
    return int(np.clip(round(score), 0, 1000))
