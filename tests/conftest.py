"""Shared fixtures: a small synthetic dataset and a quickly-trained model bundle."""
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import pytest

from credit_risk import data, evaluate, features
from credit_risk.model import build_pipeline, get_experiment
from credit_risk.train import training_profile


@pytest.fixture(scope="session")
def raw_df():
    return data.generate_synthetic(4000, seed=7)


@pytest.fixture(scope="session")
def dataset(raw_df):
    return features.prepare_dataset(raw_df)


@pytest.fixture(scope="session")
def fitted(dataset):
    X, y, _ = dataset
    spec = get_experiment("v5_hgb_tuned")
    pipe = build_pipeline(spec).fit(X, y)
    return spec, pipe


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory, dataset, fitted):
    X, y, _ = dataset
    spec, pipe = fitted
    p = pipe.predict_proba(X)[:, 1]
    thr = evaluate.select_threshold(y, p)
    metrics = evaluate.classification_metrics(y, p, thr)
    bundle = {"pipeline": pipe, "experiment": spec.name, "model_type": spec.model_type,
              "feature_columns": list(X.columns), "decline_threshold": thr, "review_threshold": thr / 2,
              "background": X.sample(100, random_state=0), "training_profile": training_profile(X),
              "metrics": {"validation": metrics, "test": metrics}, "mlflow_run_id": "test",
              "data_source": "synthetic", "version": "test", "trained_at": "2026-01-01T00:00:00+00:00"}
    path = tmp_path_factory.mktemp("model") / "bundle.joblib"
    joblib.dump(bundle, path)
    return path


@pytest.fixture
def applicant():
    return {"contract_type": "Cash loans", "gender": "F", "own_car": False, "own_realty": True,
            "children": 1, "income": 180000, "credit_amount": 500000, "annuity": 25000,
            "goods_price": 450000, "income_type": "Working", "education": "Higher education",
            "family_status": "Married", "housing_type": "House / apartment", "age_years": 38,
            "years_employed": 6.5, "occupation": "Core staff", "family_members": 3,
            "region_rating": 2, "ext_source_1": None, "ext_source_2": 0.62, "ext_source_3": 0.55}
