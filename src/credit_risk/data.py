"""Data loading and synthetic sample generation.

Real data
---------
The project is designed for the Kaggle *Home Credit Default Risk* dataset
(`application_train.csv`, ~307k applicants).  Kaggle requires a login and
acceptance of the competition rules, so the file is not committed; see
README.md for download instructions.

Sample data
-----------
`generate_synthetic()` produces data with the *same raw schema and units*
(negative DAYS_* columns, the 365243 employment sentinel, realistic missing
rates) so every part of the pipeline, the API and the tests can run without
the real file.  Results obtained on synthetic data are illustrative only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def load_raw(path: str | Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Load raw applications (Home Credit schema).

    Resolution order: explicit `path` -> data/raw/application_train.csv ->
    data/sample/application_sample.csv.
    """
    if path is None:
        path = config.RAW_DATA_PATH if config.RAW_DATA_PATH.exists() else config.SAMPLE_DATA_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download Home Credit data (see README) or run "
            "`python -m credit_risk.data --rows 20000` to create a synthetic sample."
        )
    df = pd.read_csv(path, usecols=lambda c: c in config.RAW_COLUMNS, nrows=nrows)
    missing = set(config.RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")
    return df[config.RAW_COLUMNS]


def is_synthetic(df: pd.DataFrame) -> bool:
    """Synthetic rows use IDs >= 900000 (real Home Credit IDs are < 460000)."""
    return bool((df[config.ID_COL] >= 900_000).all())


def _choice(rng, options: dict[str, float], n: int) -> np.ndarray:
    keys = list(options)
    p = np.array(list(options.values()), dtype=float)
    return rng.choice(keys, size=n, p=p / p.sum())


def generate_synthetic(n: int = 20_000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Generate synthetic applications that mimic Home Credit distributions.

    The default probability is produced by a known logistic model, which makes
    the data useful for checking that explainability methods recover the
    true drivers (external scores, debt-service ratio, age, employment...).
    """
    rng = np.random.default_rng(seed)

    gender = _choice(rng, {"F": 0.66, "M": 0.34}, n)
    age = np.clip(rng.normal(43, 12, n), 21, 69)
    education = _choice(rng, {
        "Secondary / secondary special": 0.71, "Higher education": 0.24,
        "Incomplete higher": 0.033, "Lower secondary": 0.012, "Academic degree": 0.005,
    }, n)
    income_type = _choice(rng, {
        "Working": 0.52, "Commercial associate": 0.23, "Pensioner": 0.18,
        "State servant": 0.07,
    }, n)
    income_type = np.where(age > 60, np.where(rng.random(n) < 0.7, "Pensioner", income_type), income_type)
    pensioner = income_type == "Pensioner"

    edu_income_mult = np.select(
        [education == "Higher education", education == "Academic degree", education == "Lower secondary"],
        [1.35, 1.6, 0.75], 1.0,
    )
    income = np.round(np.exp(rng.normal(11.95, 0.5, n)) * edu_income_mult * np.where(gender == "M", 1.15, 1.0), -2)
    credit = np.round(income * np.exp(rng.normal(1.0, 0.6, n)), -3).clip(45_000, 4_000_000)
    n_payments = rng.choice([12, 18, 24, 36, 48, 60], size=n, p=[0.08, 0.1, 0.25, 0.3, 0.17, 0.1])
    annuity = np.round(credit / n_payments * rng.uniform(1.05, 1.35, n), 1)
    contract = np.where(rng.random(n) < 0.095, "Revolving loans", "Cash loans")
    goods = np.where(contract == "Revolving loans", np.nan, np.round(credit * rng.uniform(0.75, 1.0, n), -3))

    years_emp = np.clip(rng.exponential(6.5, n), 0, age - 18)
    days_employed = np.where(pensioner & (rng.random(n) < 0.98), config.DAYS_EMPLOYED_SENTINEL,
                             -np.round(years_emp * 365.25))
    children = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.7, 0.2, 0.087, 0.012, 0.001])
    family_status = _choice(rng, {
        "Married": 0.64, "Single / not married": 0.15, "Civil marriage": 0.1,
        "Separated": 0.06, "Widow": 0.05,
    }, n)
    fam_members = children + np.where(np.isin(family_status, ["Married", "Civil marriage"]), 2, 1)
    housing = _choice(rng, {
        "House / apartment": 0.89, "With parents": 0.05, "Municipal apartment": 0.035,
        "Rented apartment": 0.016, "Office apartment": 0.009,
    }, n)
    occupation = _choice(rng, {
        "Laborers": 0.26, "Sales staff": 0.15, "Core staff": 0.13, "Managers": 0.1,
        "Drivers": 0.09, "High skill tech staff": 0.054, "Accountants": 0.046,
        "Medicine staff": 0.04, "Security staff": 0.032, "Cooking staff": 0.028,
        "Cleaning staff": 0.022, "Low-skill Laborers": 0.01,
    }, n).astype(object)
    occupation[pensioner | (rng.random(n) < 0.12)] = np.nan
    region = rng.choice([1, 2, 3], size=n, p=[0.1, 0.74, 0.16])
    own_car = np.where(rng.random(n) < np.where(gender == "M", 0.55, 0.25), "Y", "N")
    own_realty = np.where(rng.random(n) < 0.69, "Y", "N")

    # Latent creditworthiness drives the three external bureau scores.
    latent = rng.normal(0, 1, n) + 0.02 * (age - 43) + 0.25 * (education == "Higher education")
    def ext(noise, missing_rate):
        s = 1 / (1 + np.exp(-(0.9 * latent + rng.normal(0, noise, n))))
        s[rng.random(n) < missing_rate] = np.nan
        return np.round(s, 6)
    ext1, ext2, ext3 = ext(0.9, 0.56), ext(0.7, 0.002), ext(0.8, 0.2)

    # "True" default model (log-odds)
    dsti = annuity / income
    logit = (
        -2.75
        - 1.05 * latent
        + 2.2 * (dsti - 0.18)
        - 0.012 * (age - 43)
        - 0.03 * np.where(pensioner, 5, years_emp)
        + 0.35 * (education == "Lower secondary")
        - 0.30 * np.isin(education, ["Higher education", "Academic degree"])
        + 0.25 * (region == 3) - 0.2 * (region == 1)
        + 0.25 * np.isin(occupation.astype(str), ["Laborers", "Drivers", "Low-skill Laborers"])
        + 0.20 * (gender == "M")              # historical correlation present in real data
        + 0.15 * (housing == "Rented apartment") + 0.12 * (housing == "With parents")
        - 0.25 * (contract == "Revolving loans")
        + 0.08 * children
    )
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        config.ID_COL: np.arange(900_000, 900_000 + n),
        config.RAW_TARGET: target,
        "NAME_CONTRACT_TYPE": contract,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": children,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing,
        "DAYS_BIRTH": -np.round(age * 365.25).astype(int),
        "DAYS_EMPLOYED": days_employed.astype(int),
        "OCCUPATION_TYPE": occupation,
        "CNT_FAM_MEMBERS": fam_members.astype(float),
        "REGION_RATING_CLIENT": region,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
    })
    return df[config.RAW_COLUMNS]


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Generate the synthetic sample dataset.")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--out", type=Path, default=config.SAMPLE_DATA_PATH)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic(args.rows, args.seed)
    data.to_csv(args.out, index=False)
    print(f"Wrote {len(data):,} rows to {args.out} (default rate {data[config.RAW_TARGET].mean():.2%})")
