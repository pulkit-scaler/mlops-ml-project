"""The three data decisions are load bearing, so they are pinned by tests."""
import numpy as np
import pandas as pd

from campaign.data import FEATURE_ORDER, LEAKING, NEVER_CONTACTED, clean


def test_duration_is_present_in_the_raw_file(raw):
    """If this ever fails the dataset changed, and the leakage lesson with it."""
    assert LEAKING in raw.columns


def test_duration_never_reaches_the_model(cleaned):
    X, _ = cleaned
    assert LEAKING not in X.columns


def test_pdays_sentinel_is_the_overwhelming_majority(raw):
    share = (raw["pdays"] == NEVER_CONTACTED).mean()
    assert 0.95 < share < 0.97


def test_pdays_sentinel_becomes_a_gap_and_a_flag(cleaned):
    X, _ = cleaned
    assert NEVER_CONTACTED not in X["pdays"].dropna().values
    never = X["previously_contacted"] == 0
    assert X.loc[never, "pdays"].isna().all()
    assert X.loc[~never, "pdays"].notna().all()


def test_columns_are_exactly_the_agreed_order(cleaned):
    X, _ = cleaned
    assert list(X.columns) == FEATURE_ORDER


def test_target_is_binary_and_imbalanced(cleaned):
    _, y = cleaned
    assert set(y.unique()) == {0, 1}
    assert 0.10 < y.mean() < 0.12
