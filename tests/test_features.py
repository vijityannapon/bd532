import numpy as np
import pandas as pd

from credit_risk import config, features


def test_raw_conversion_units_and_sentinel(raw_df):
    apps = features.raw_to_applicants(raw_df)
    assert apps["age_years"].between(18, 100).all()
    sentinel = raw_df["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_SENTINEL
    assert sentinel.any()
    assert apps.loc[sentinel, "years_employed"].isna().all()          # 365243 -> NaN
    assert (apps.loc[~sentinel, "years_employed"] >= 0).all()
    assert set(apps["own_car"].unique()) <= {0, 1}


def test_sensitive_attributes_never_in_features(dataset):
    X, _, s = dataset
    assert "gender" not in X.columns and "age_group" not in X.columns
    assert list(s.columns) == ["gender", "age_group"]


def test_engineered_ratios():
    df = pd.DataFrame([{"income": 100.0, "credit_amount": 400.0, "annuity": 20.0, "goods_price": 360.0,
                        "years_employed": 10.0, "age_years": 40.0,
                        "ext_source_1": np.nan, "ext_source_2": 0.4, "ext_source_3": 0.6}])
    out = features.add_engineered_features(df).iloc[0]
    assert out.credit_income_ratio == 4.0
    assert out.annuity_income_ratio == 0.2
    assert out.payment_rate == 0.05
    assert out.goods_credit_ratio == 0.9
    assert out.employed_age_ratio == 0.25
    assert out.ext_source_mean == 0.5 and out.ext_source_missing == 1


def test_zero_income_does_not_produce_inf():
    df = pd.DataFrame([{"income": 0.0, "credit_amount": 0.0, "annuity": 1.0, "goods_price": 1.0,
                        "years_employed": 1.0, "age_years": 30.0,
                        "ext_source_1": 0.1, "ext_source_2": 0.1, "ext_source_3": 0.1}])
    out = features.add_engineered_features(df)
    assert not np.isinf(out.select_dtypes("number").to_numpy()).any()


def test_cleaning_invalid_ranges(applicant):
    a = dict(applicant, ext_source_2=1.7, years_employed=-3)
    X = features.applicants_to_features(pd.DataFrame([a]))
    assert np.isnan(X.loc[0, "ext_source_2"]) and np.isnan(X.loc[0, "years_employed"])


def test_feature_column_sets():
    assert set(features.feature_columns(False)) < set(features.feature_columns(True))
