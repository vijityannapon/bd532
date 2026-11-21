"""Central configuration: paths, feature schema, and business policy constants.

Everything that more than one module needs to agree on lives here, so the
training pipeline, the API and the tests share a single source of truth.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "application_train.csv"      # Home Credit (Kaggle)
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "application_sample.csv"  # synthetic, same schema
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_BUNDLE_PATH = MODELS_DIR / "credit_risk_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
FEEDBACK_PATH = PROJECT_ROOT / "data" / "feedback" / "feedback.jsonl"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "bd532-credit-risk"

RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Raw schema (subset of Home Credit `application_train.csv`)
# --------------------------------------------------------------------------- #
ID_COL = "SK_ID_CURR"
RAW_TARGET = "TARGET"  # 1 = client had payment difficulties (default), 0 = repaid

# Raw Home Credit column -> canonical "applicant" field used everywhere else
RAW_TO_APPLICANT = {
    "NAME_CONTRACT_TYPE": "contract_type",
    "CODE_GENDER": "gender",
    "FLAG_OWN_CAR": "own_car",
    "FLAG_OWN_REALTY": "own_realty",
    "CNT_CHILDREN": "children",
    "AMT_INCOME_TOTAL": "income",
    "AMT_CREDIT": "credit_amount",
    "AMT_ANNUITY": "annuity",
    "AMT_GOODS_PRICE": "goods_price",
    "NAME_INCOME_TYPE": "income_type",
    "NAME_EDUCATION_TYPE": "education",
    "NAME_FAMILY_STATUS": "family_status",
    "NAME_HOUSING_TYPE": "housing_type",
    "DAYS_BIRTH": "age_years",          # converted: -days / 365.25
    "DAYS_EMPLOYED": "years_employed",  # converted; 365243 sentinel -> NaN
    "OCCUPATION_TYPE": "occupation",
    "CNT_FAM_MEMBERS": "family_members",
    "REGION_RATING_CLIENT": "region_rating",
    "EXT_SOURCE_1": "ext_source_1",
    "EXT_SOURCE_2": "ext_source_2",
    "EXT_SOURCE_3": "ext_source_3",
}
RAW_COLUMNS = [ID_COL, RAW_TARGET, *RAW_TO_APPLICANT.keys()]
DAYS_EMPLOYED_SENTINEL = 365243  # Home Credit code for "not employed / pensioner"

TARGET = "default"

# --------------------------------------------------------------------------- #
# Model features
# --------------------------------------------------------------------------- #
CATEGORICAL_FEATURES = [
    "contract_type",
    "income_type",
    "education",
    "family_status",
    "housing_type",
    "occupation",
]
BASE_NUMERIC_FEATURES = [
    "own_car",
    "own_realty",
    "children",
    "income",
    "credit_amount",
    "annuity",
    "goods_price",
    "age_years",
    "years_employed",
    "family_members",
    "region_rating",
    "ext_source_1",
    "ext_source_2",
    "ext_source_3",
]
ENGINEERED_FEATURES = [
    "credit_income_ratio",    # loan size relative to annual income
    "annuity_income_ratio",   # debt-service-to-income (DSTI)
    "payment_rate",           # annuity / credit  (~ 1 / number of payments)
    "goods_credit_ratio",     # goods price / credit  (~ loan-to-value proxy)
    "employed_age_ratio",     # share of life in current employment
    "employment_missing",     # 1 if pensioner / unemployed sentinel
    "ext_source_mean",
    "ext_source_missing",     # number of missing external scores
]

# Sensitive attributes: NEVER fed to the model, only used for fairness audits.
SENSITIVE_ATTRIBUTES = ["gender", "age_group"]
AGE_BINS = [0, 30, 50, 200]
AGE_LABELS = ["<30", "30-50", ">50"]

# --------------------------------------------------------------------------- #
# Business policy (used for thresholding and decision bands)
# --------------------------------------------------------------------------- #
# Expected cost of a missed defaulter (approve a bad loan) relative to the cost
# of rejecting a good customer (lost margin).  5:1 is a common rule of thumb
# for unsecured consumer credit and is documented in docs/02_system_design.md.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0
REVIEW_BAND_FACTOR = 0.5  # review band starts at REVIEW_BAND_FACTOR * decline threshold

MODEL_VERSION_TAG = "1.0.0"
