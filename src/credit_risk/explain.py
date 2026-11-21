"""Explainability (Section 3.4) and per-prediction reasoning (Section 3.5).

* Global: SHAP mean |value| per applicant field + beeswarm plot, and
  permutation importance (model-agnostic cross-check).
* Local: `explain_applicant()` turns SHAP values into ranked, plain-language
  reasons ("Debt-service ratio 45% of income -> increases risk").

SHAP values for one-hot columns are summed back to the original field so a
loan officer sees "Education" rather than "education_Higher education".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from .model import transformed_feature_map

# Human-readable labels (EN, TH) and value formatters for every model field.
FEATURE_LABELS: dict[str, tuple[str, str]] = {
    "contract_type": ("Contract type", "ประเภทสัญญา"),
    "income_type": ("Income type", "ประเภทรายได้"),
    "education": ("Education", "ระดับการศึกษา"),
    "family_status": ("Family status", "สถานภาพครอบครัว"),
    "housing_type": ("Housing type", "ลักษณะที่อยู่อาศัย"),
    "occupation": ("Occupation", "อาชีพ"),
    "own_car": ("Owns a car", "มีรถยนต์"),
    "own_realty": ("Owns real estate", "มีอสังหาริมทรัพย์"),
    "children": ("Number of children", "จำนวนบุตร"),
    "income": ("Annual income", "รายได้ต่อปี"),
    "credit_amount": ("Loan amount", "วงเงินกู้"),
    "annuity": ("Loan annuity (installment)", "ค่างวด"),
    "goods_price": ("Price of goods / collateral", "ราคาสินทรัพย์ที่ซื้อ"),
    "age_years": ("Age", "อายุ"),
    "years_employed": ("Years in current job", "อายุงาน"),
    "family_members": ("Family members", "จำนวนสมาชิกครอบครัว"),
    "region_rating": ("Region risk rating (1-3)", "ระดับความเสี่ยงพื้นที่ (1-3)"),
    "ext_source_1": ("External credit score 1", "คะแนนเครดิตภายนอก 1"),
    "ext_source_2": ("External credit score 2", "คะแนนเครดิตภายนอก 2"),
    "ext_source_3": ("External credit score 3", "คะแนนเครดิตภายนอก 3"),
    "credit_income_ratio": ("Loan-to-income ratio", "สัดส่วนวงเงินกู้ต่อรายได้"),
    "annuity_income_ratio": ("Debt-service-to-income ratio", "สัดส่วนค่างวดต่อรายได้ (DSTI)"),
    "payment_rate": ("Payment rate (installment / loan)", "อัตราผ่อนต่อวงเงิน"),
    "goods_credit_ratio": ("Goods price / loan ratio", "สัดส่วนราคาสินทรัพย์ต่อวงเงิน"),
    "employed_age_ratio": ("Share of life employed", "สัดส่วนอายุงานต่ออายุ"),
    "employment_missing": ("No current employment record", "ไม่มีข้อมูลการจ้างงานปัจจุบัน"),
    "ext_source_mean": ("Average external credit score", "คะแนนเครดิตภายนอกเฉลี่ย"),
    "ext_source_missing": ("Missing external scores (count)", "จำนวนคะแนนภายนอกที่ไม่มีข้อมูล"),
}

_PERCENT = {"annuity_income_ratio", "employed_age_ratio"}
_MONEY = {"income", "credit_amount", "annuity", "goods_price"}
_BOOL = {"own_car", "own_realty", "employment_missing"}


def format_value(feature: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    if feature in _BOOL:
        return "yes" if float(value) >= 0.5 else "no"
    if feature in _MONEY:
        return f"{float(value):,.0f}"
    if feature in _PERCENT:
        return f"{float(value):.0%}"
    if feature in {"age_years", "years_employed"}:
        return f"{float(value):.1f} yrs"
    if isinstance(value, (int, float, np.number)):
        return f"{float(value):.3g}"
    return str(value)


@dataclass
class Reason:
    feature: str
    label: str
    label_th: str
    value: str
    shap_value: float      # contribution in log-odds of default
    direction: str         # "increases_risk" | "decreases_risk"
    text: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class CreditExplainer:
    """Wraps a fitted pipeline with the right SHAP explainer for its model type."""

    def __init__(self, pipeline, background: pd.DataFrame | None = None):
        self.pipeline = pipeline
        self.prep = pipeline.named_steps["prep"]
        self.clf = pipeline.named_steps["clf"]
        self.feature_map = np.array(transformed_feature_map(pipeline))
        self.original_features = list(dict.fromkeys(self.feature_map))
        name = type(self.clf).__name__
        if name in {"HistGradientBoostingClassifier", "RandomForestClassifier"}:
            self.explainer = shap.TreeExplainer(self.clf)
        elif name == "LogisticRegression":
            if background is None:
                raise ValueError("LinearExplainer needs background data")
            self.explainer = shap.LinearExplainer(self.clf, self.prep.transform(background))
        else:  # pragma: no cover
            raise ValueError(f"No explainer configured for {name}")

    def _raw_shap(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        Xt = self.prep.transform(X)
        sv = self.explainer.shap_values(Xt)
        base = self.explainer.expected_value
        if isinstance(sv, list):              # older SHAP: [class0, class1]
            sv, base = sv[1], base[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                      # (n, features, classes)
            sv = sv[:, :, 1]
            base = np.asarray(base).ravel()[-1]
        return sv, float(np.asarray(base).ravel()[-1])

    def shap_by_feature(self, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """SHAP values aggregated to original applicant fields (log-odds units)."""
        sv, base = self._raw_shap(X)
        agg = pd.DataFrame(sv, columns=self.feature_map, index=X.index)
        agg = agg.T.groupby(level=0, sort=False).sum().T
        return agg[self.original_features], base

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        agg, _ = self.shap_by_feature(X)
        return agg.abs().mean().sort_values(ascending=False)

    def explain_applicant(self, X_row: pd.DataFrame, top_k: int = 5) -> dict:
        """Plain-language explanation for a single applicant."""
        assert len(X_row) == 1, "explain_applicant expects exactly one row"
        agg, base = self.shap_by_feature(X_row)
        contrib = agg.iloc[0].sort_values(key=np.abs, ascending=False)
        reasons = []
        for feat, val in contrib.head(top_k).items():
            raw_val = X_row.iloc[0][feat]
            label, label_th = FEATURE_LABELS.get(feat, (feat, feat))
            direction = "increases_risk" if val > 0 else "decreases_risk"
            verb = "increases" if val > 0 else "decreases"
            vtxt = format_value(feat, raw_val)
            reasons.append(Reason(
                feature=feat, label=label, label_th=label_th, value=vtxt,
                shap_value=round(float(val), 4), direction=direction,
                text=f"{label} = {vtxt} {verb} default risk",
            ).as_dict())
        logit = base + float(agg.iloc[0].sum())
        return {
            "base_log_odds": round(base, 4),
            "base_probability": round(float(1 / (1 + np.exp(-base))), 4),
            "explained_probability": round(float(1 / (1 + np.exp(-logit))), 4),
            "reasons": reasons,
        }


def plot_global(explainer: CreditExplainer, X: pd.DataFrame, out_dir) -> list[str]:
    """Save SHAP bar + beeswarm plots (aggregated to applicant fields)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg, base = explainer.shap_by_feature(X)
    # For colouring the beeswarm use the numeric value when available.
    disp = pd.DataFrame(index=X.index)
    for f in agg.columns:
        disp[f] = pd.to_numeric(X[f], errors="coerce") if f in X else np.nan
    exp = shap.Explanation(values=agg.to_numpy(), base_values=np.full(len(agg), base),
                           data=disp.to_numpy(), feature_names=list(agg.columns))
    paths = []
    for kind in ("bar", "beeswarm"):
        plt.figure()
        getattr(shap.plots, kind)(exp, max_display=15, show=False)
        p = f"{out_dir}/shap_{kind}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=130, bbox_inches="tight")
        plt.close("all")
        paths.append(p)
    return paths
