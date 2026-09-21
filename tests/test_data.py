"""The cleaning decisions are load bearing, so they are pinned by tests."""
import pandas as pd

from churn.data import FEATURE_ORDER, IDENTIFIER


def test_raw_totalcharges_is_text_and_that_is_the_bug(raw):
    assert raw["TotalCharges"].dtype == object
    blanks = raw["TotalCharges"].astype(str).str.strip() == ""
    assert blanks.sum() == 11


def test_the_eleven_blanks_are_all_brand_new_customers(raw):
    blanks = raw["TotalCharges"].astype(str).str.strip() == ""
    assert (raw.loc[blanks, "tenure"] == 0).all()


def test_clean_produces_numeric_charges_and_no_identifier(cleaned):
    X, y = cleaned
    assert pd.api.types.is_numeric_dtype(X["TotalCharges"])
    assert IDENTIFIER not in X.columns
    assert list(X.columns) == FEATURE_ORDER
    assert X["TotalCharges"].isna().sum() == 0


def test_new_customers_are_charged_zero_not_the_median(cleaned):
    X, _ = cleaned
    assert (X.loc[X["tenure"] == 0, "TotalCharges"] == 0).all()


def test_target_is_binary_and_imbalanced(cleaned):
    _, y = cleaned
    assert set(y.unique()) == {0, 1}
    assert 0.25 < y.mean() < 0.28
