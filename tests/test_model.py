from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from credit_risk import evaluate, fairness
from credit_risk.explain import CreditExplainer
from credit_risk.features import feature_columns
from credit_risk.model import EXPERIMENTS, build_pipeline, reweighing_weights, transformed_feature_map


def test_at_least_three_experiments():
    assert len(EXPERIMENTS) >= 3
    assert len({e.name for e in EXPERIMENTS}) == len(EXPERIMENTS)


@pytest.mark.parametrize("spec", EXPERIMENTS[:4], ids=lambda s: s.name)
def test_every_model_trains_and_outputs_probabilities(spec, dataset):
    X, y, _ = dataset
    cols = feature_columns(spec.engineered)
    if spec.model_type == "random_forest":  # keep the test fast without mutating the shared spec
        spec = replace(spec, params={**spec.params, "n_estimators": 20})
    pipe = build_pipeline(spec).fit(X[cols].iloc[:1500], y.iloc[:1500])
    p = pipe.predict_proba(X[cols].iloc[1500:2000])[:, 1]
    assert p.min() >= 0 and p.max() <= 1


def test_model_beats_random(dataset, fitted):
    from sklearn.metrics import roc_auc_score
    X, y, _ = dataset
    _, pipe = fitted
    assert roc_auc_score(y, pipe.predict_proba(X)[:, 1]) > 0.65


def test_unseen_category_is_handled(fitted, dataset):
    X, _, _ = dataset
    _, pipe = fitted
    row = X.iloc[[0]].copy()
    row["occupation"] = "Astronaut"
    assert 0 <= pipe.predict_proba(row)[0, 1] <= 1


def test_feature_map_matches_transformed_width(fitted, dataset):
    X, _, _ = dataset
    _, pipe = fitted
    fmap = transformed_feature_map(pipe)
    assert len(fmap) == pipe.named_steps["prep"].transform(X.head(2)).shape[1]
    assert set(fmap) == set(X.columns)


def test_shap_values_add_up_to_prediction(fitted, dataset):
    X, _, _ = dataset
    _, pipe = fitted
    expl = CreditExplainer(pipe)
    row = X.iloc[[3]]
    out = expl.explain_applicant(row, top_k=3)
    assert len(out["reasons"]) == 3
    assert out["explained_probability"] == pytest.approx(pipe.predict_proba(row)[0, 1], abs=1e-3)


def test_reweighing_removes_group_target_dependence():
    y = np.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
    g = np.array(["M"] * 4 + ["F"] * 6)
    w = reweighing_weights(y, g)
    df = pd.DataFrame({"y": y, "g": g, "w": w})
    rates = df.groupby("g").apply(lambda d: np.average(d.y, weights=d.w), include_groups=False)
    assert rates["M"] == pytest.approx(rates["F"]) == pytest.approx(y.mean())


def test_metrics_and_threshold():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    assert evaluate.ks_statistic(y, p) == 1.0
    thr = evaluate.select_threshold(y, p)
    assert 0.4 < thr <= 0.8
    m = evaluate.classification_metrics(y, p, thr)
    assert m["recall"] == 1.0 and m["fp"] == 0


def test_decision_bands_and_score_monotonic():
    assert evaluate.decision_band(0.30, 0.2) == "HIGH_RISK"
    assert evaluate.decision_band(0.12, 0.2) == "REVIEW"
    assert evaluate.decision_band(0.05, 0.2) == "LOW_RISK"
    scores = [evaluate.risk_score(p) for p in (0.01, 0.05, 0.2, 0.6)]
    assert scores == sorted(scores, reverse=True)


def test_fairness_report():
    y = np.array([1, 0, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.9, 0.1, 0.1])
    g = np.array(["A", "A", "A", "B", "B", "B"])
    res = fairness.audit(y, p, pd.DataFrame({"grp": g}), threshold=0.5)["grp"]
    assert res["by_group"].loc["A", "approval_rate"] == pytest.approx(1 / 3)
    assert res["by_group"].loc["B", "approval_rate"] == pytest.approx(2 / 3)
    assert res["summary"]["disparate_impact_ratio"] == pytest.approx(0.5)
    assert res["summary"]["fpr_diff"] == pytest.approx(0.5)
