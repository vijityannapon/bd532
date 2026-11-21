"""Model service: loads the exported bundle once and serves predictions + explanations."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_risk import config, evaluate
from credit_risk.explain import CreditExplainer
from credit_risk.features import applicants_to_features

RECOMMENDATIONS = {
    "LOW_RISK": "Low risk - proceed with standard approval checks.",
    "REVIEW": "Moderate risk - verify income/obligations before deciding.",
    "HIGH_RISK": "High risk - senior officer review required; consider lower amount or collateral.",
}


class BusinessRuleError(ValueError):
    """Input is well-formed but logically inconsistent (HTTP 400)."""

    def __init__(self, message: str, details: list | None = None):
        super().__init__(message)
        self.details = details or []


def check_business_rules(a: dict) -> None:
    errs = []
    if a["annuity"] > a["credit_amount"]:
        errs.append({"field": "annuity", "issue": "installment cannot exceed the loan amount"})
    if a.get("years_employed") is not None and a["years_employed"] > a["age_years"] - 14:
        errs.append({"field": "years_employed", "issue": "employment longer than working age"})
    if a["family_members"] < a["children"] + 1:
        errs.append({"field": "family_members", "issue": "must be >= children + 1"})
    if errs:
        raise BusinessRuleError("Application data is inconsistent", errs)


class ModelService:
    def __init__(self, bundle_path: str | Path | None = None):
        path = Path(bundle_path or os.getenv("MODEL_PATH", config.MODEL_BUNDLE_PATH))
        self.bundle = joblib.load(path)
        self.pipeline = self.bundle["pipeline"]
        self.columns = self.bundle["feature_columns"]
        self.decline_threshold = float(self.bundle["decline_threshold"])
        self.review_threshold = float(self.bundle["review_threshold"])
        self.explainer = CreditExplainer(self.pipeline, self.bundle.get("background"))
        self.profile = self.bundle.get("training_profile", {})

    # ------------------------------------------------------------------ #
    @property
    def info(self) -> dict:
        b = self.bundle
        return {"name": b["experiment"], "version": b["version"], "trained_at": b["trained_at"],
                "data_source": b["data_source"]}

    def _out_of_range(self, X: pd.DataFrame) -> list[str]:
        warnings = []
        for col, s in self.profile.get("numeric", {}).items():
            v = X.iloc[0].get(col)
            if v is None or pd.isna(v):
                continue
            if v < s["p01"] or v > s["p99"]:
                warnings.append(f"{col}={v:.4g} is outside the training range "
                                f"[{s['p01']:.4g}, {s['p99']:.4g}] - prediction is less reliable")
        return warnings

    def predict(self, applicant: dict, explain: bool = True, top_k: int = 5) -> dict:
        check_business_rules(applicant)
        X = applicants_to_features(pd.DataFrame([applicant]))[self.columns]
        pd_score = float(self.pipeline.predict_proba(X)[0, 1])
        band = evaluate.decision_band(pd_score, self.decline_threshold)
        result = {
            "prediction": {
                "probability_of_default": round(pd_score, 4),
                "risk_score": evaluate.risk_score(pd_score),
                "decision_band": band,
                "recommendation": RECOMMENDATIONS[band],
                "thresholds": {"review": round(self.review_threshold, 4),
                               "high_risk": round(self.decline_threshold, 4)},
            },
            "warnings": self._out_of_range(X),
        }
        if explain:
            e = self.explainer.explain_applicant(X, top_k=top_k)
            result["explanation"] = {"base_probability": e["base_probability"], "reasons": e["reasons"]}
        return result

    def metadata(self) -> dict:
        b = self.bundle
        def clean(d):
            return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}
        return {**self.info, "model_type": b["model_type"], "mlflow_run_id": b["mlflow_run_id"],
                "features": self.columns,
                "thresholds": {"review": self.review_threshold, "high_risk": self.decline_threshold},
                "metrics": {"validation": clean(b["metrics"]["validation"]), "test": clean(b["metrics"]["test"]),
                            "cv_roc_auc_mean": b["metrics"].get("cv_roc_auc_mean"),
                            "cv_roc_auc_std": b["metrics"].get("cv_roc_auc_std")}}
