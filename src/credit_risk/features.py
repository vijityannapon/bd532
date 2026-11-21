"""Data preprocessing: raw Home Credit rows -> canonical applicant frame -> model features.

Two entry points are used by the rest of the system:

* `prepare_dataset(raw_df)` -> (X, y, sensitive) for training / evaluation.
* `applicants_to_features(applicant_df)` for the API, which receives
  canonical applicant fields (human units) directly from the UI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def raw_to_applicants(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns and convert units into human-readable applicant fields."""
    df = raw.rename(columns=config.RAW_TO_APPLICANT).copy()

    # Days are stored as negative offsets from the application date.
    df["age_years"] = -df["age_years"] / 365.25
    emp = df["years_employed"].astype(float)
    df["years_employed"] = np.where(emp == config.DAYS_EMPLOYED_SENTINEL, np.nan, -emp / 365.25)

    for col in ("own_car", "own_realty"):
        df[col] = (df[col].astype(str).str.upper() == "Y").astype(int)

    # Home Credit encodes 4 rows with gender 'XNA' - treat as unknown.
    df["gender"] = df["gender"].where(df["gender"].isin(["M", "F"]), other=np.nan)
    return df


def clean_applicants(df: pd.DataFrame) -> pd.DataFrame:
    """Type coercion and range sanitisation shared by training and serving."""
    df = df.copy()
    for col in config.BASE_NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in config.CATEGORICAL_FEATURES:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    # Negative employment or impossible ages are data errors -> missing.
    df.loc[df["years_employed"] < 0, "years_employed"] = np.nan
    df.loc[(df["age_years"] < 18) | (df["age_years"] > 100), "age_years"] = np.nan
    for col in ("ext_source_1", "ext_source_2", "ext_source_3"):
        df.loc[(df[col] < 0) | (df[col] > 1), col] = np.nan
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain ratios used by credit analysts (affordability, leverage, stability)."""
    df = df.copy()
    income = df["income"].replace(0, np.nan)
    credit = df["credit_amount"].replace(0, np.nan)
    df["credit_income_ratio"] = df["credit_amount"] / income
    df["annuity_income_ratio"] = df["annuity"] / income
    df["payment_rate"] = df["annuity"] / credit
    df["goods_credit_ratio"] = df["goods_price"] / credit
    df["employed_age_ratio"] = df["years_employed"] / df["age_years"]
    df["employment_missing"] = df["years_employed"].isna().astype(int)
    ext = df[["ext_source_1", "ext_source_2", "ext_source_3"]]
    df["ext_source_mean"] = ext.mean(axis=1)
    df["ext_source_missing"] = ext.isna().sum(axis=1)
    return df.replace([np.inf, -np.inf], np.nan)


def age_group(age_years: pd.Series) -> pd.Series:
    return pd.cut(age_years, bins=config.AGE_BINS, labels=config.AGE_LABELS, right=False).astype(object)


def feature_columns(engineered: bool = True) -> list[str]:
    cols = config.CATEGORICAL_FEATURES + config.BASE_NUMERIC_FEATURES
    return cols + config.ENGINEERED_FEATURES if engineered else cols


def applicants_to_features(applicants: pd.DataFrame, engineered: bool = True) -> pd.DataFrame:
    """Canonical applicant rows -> model input frame (sensitive attributes dropped)."""
    df = add_engineered_features(clean_applicants(applicants))
    return df[feature_columns(engineered)]


def prepare_dataset(raw: pd.DataFrame, engineered: bool = True):
    """Return (X, y, sensitive) where `sensitive` is kept aside for fairness audits."""
    apps = raw_to_applicants(raw)
    X = applicants_to_features(apps, engineered=engineered)
    y = apps[config.RAW_TARGET].astype(int).rename(config.TARGET)
    sensitive = pd.DataFrame({
        "gender": apps["gender"],
        "age_group": age_group(apps["age_years"]),
    }, index=apps.index)
    return X, y, sensitive
