"""Model architecture definitions.

Each experiment is an sklearn `Pipeline`:

    ColumnTransformer
      ├─ numeric:     median imputation (+ missing indicators) [+ scaling for linear models]
      └─ categorical: constant "Unknown" imputation -> one-hot (rare levels grouped)
    -> classifier (LogisticRegression | RandomForest | HistGradientBoosting)

Keeping preprocessing inside the pipeline guarantees that training and the
REST API apply *identical* transformations (no training/serving skew).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .features import feature_columns


@dataclass
class ExperimentSpec:
    """One model version tracked in MLflow."""

    name: str
    model_type: str                 # "logreg" | "random_forest" | "hist_gb"
    engineered: bool = True         # use domain ratio features?
    params: dict = field(default_factory=dict)
    reweigh: bool = False           # fairness mitigation: Kamiran-Calders reweighing
    description: str = ""


EXPERIMENTS: list[ExperimentSpec] = [
    ExperimentSpec(
        "v1_logreg_basic", "logreg", engineered=False,
        params={"C": 1.0, "class_weight": None},
        description="Baseline: logistic regression on raw fields only.",
    ),
    ExperimentSpec(
        "v2_logreg_engineered", "logreg", engineered=True,
        params={"C": 0.5, "class_weight": None},
        description="Logistic regression + domain ratio features (preprocessing change only).",
    ),
    ExperimentSpec(
        "v3_random_forest", "random_forest", engineered=True,
        params={"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 50,
                "max_features": "sqrt", "class_weight": "balanced_subsample"},
        description="Bagged trees, non-linear, robust to outliers.",
    ),
    ExperimentSpec(
        "v4_hgb_default", "hist_gb", engineered=True,
        params={"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31},
        description="Gradient boosting with library defaults.",
    ),
    ExperimentSpec(
        "v5_hgb_tuned", "hist_gb", engineered=True,
        params={"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 24,
                "min_samples_leaf": 80, "l2_regularization": 1.0,
                "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 30},
        description="Gradient boosting, slower learning rate + regularisation + early stopping.",
    ),
    ExperimentSpec(
        "v6_hgb_tuned_reweighed", "hist_gb", engineered=True,
        params={"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 24,
                "min_samples_leaf": 80, "l2_regularization": 1.0,
                "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 30},
        reweigh=True,
        description="v5 + reweighing on gender x target (fairness mitigation).",
    ),
]


def get_experiment(name: str) -> ExperimentSpec:
    for spec in EXPERIMENTS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown experiment '{name}'. Options: {[e.name for e in EXPERIMENTS]}")


def build_preprocessor(engineered: bool, scale: bool) -> ColumnTransformer:
    numeric = [c for c in feature_columns(engineered) if c not in config.CATEGORICAL_FEATURES]
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                                 sparse_output=False)),
    ]
    return ColumnTransformer(
        [("num", Pipeline(num_steps), numeric),
         ("cat", Pipeline(cat_steps), config.CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )


def build_classifier(model_type: str, params: dict):
    rs = config.RANDOM_STATE
    if model_type == "logreg":
        return LogisticRegression(max_iter=2000, random_state=rs, **params)
    if model_type == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=rs, **params)
    if model_type == "hist_gb":
        return HistGradientBoostingClassifier(random_state=rs, **params)
    raise ValueError(f"Unsupported model_type: {model_type}")


def build_pipeline(spec: ExperimentSpec) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(spec.engineered, scale=spec.model_type == "logreg")),
        ("clf", build_classifier(spec.model_type, spec.params)),
    ])


def reweighing_weights(y, group) -> np.ndarray:
    """Kamiran & Calders (2012) reweighing: w(g, y) = P(g)P(y) / P(g, y).

    Makes the target statistically independent of the sensitive group in the
    weighted training set, without using the group as a model feature.
    """
    import pandas as pd

    df = pd.DataFrame({"g": np.asarray(group, dtype=object), "y": np.asarray(y)})
    df["g"] = df["g"].fillna("Unknown")
    p_g = df["g"].value_counts(normalize=True)
    p_y = df["y"].value_counts(normalize=True)
    p_gy = df.groupby(["g", "y"]).size() / len(df)
    return df.apply(lambda r: p_g[r.g] * p_y[r.y] / p_gy[(r.g, r.y)], axis=1).to_numpy()


def transformed_feature_map(pipeline: Pipeline) -> list[str]:
    """For every column produced by the preprocessor, return the *original*
    applicant field it came from (used to aggregate SHAP values of one-hot
    columns back to a single human-readable feature)."""
    prep: ColumnTransformer = pipeline.named_steps["prep"]
    mapping: list[str] = []
    for name, trans, cols in prep.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        out = list(trans.get_feature_names_out(cols))
        if name == "num":
            for o in out:
                base = o.replace("missingindicator_", "")
                mapping.append(base if base in cols else o)
        else:
            for o in out:
                mapping.append(max((c for c in cols if o.startswith(c + "_")), key=len))
    return mapping
