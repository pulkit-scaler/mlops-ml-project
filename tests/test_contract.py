"""The request schema and the fitted encoder must agree.

This is the test that stops the two halves of the project drifting apart. If
somebody retrains on data with a new job category and does not widen the
Literal, this fails before the service accepts a value it would silently
ignore.
"""
import typing

from app.schemas import Customer
from campaign.data import CATEGORICAL_FEATURES, MARKET_CONTEXT


def allowed(field):
    return set(typing.get_args(field.annotation))


def test_every_categorical_literal_matches_the_encoder(meta):
    for column in CATEGORICAL_FEATURES:
        assert allowed(Customer.model_fields[column]) == set(meta["categories"][column]), column


def test_the_request_never_asks_for_economic_context(meta):
    """Those five columns are the service's job, not the caller's."""
    for column in MARKET_CONTEXT:
        assert column not in Customer.model_fields


def test_the_request_never_exposes_the_sentinel():
    """999 is a quirk of the CSV. The contract says null."""
    assert "pdays" not in Customer.model_fields
    assert "previously_contacted" not in Customer.model_fields
    assert "days_since_last_contact" in Customer.model_fields


def test_request_fields_plus_derived_plus_context_cover_the_model(meta):
    caller = set(Customer.model_fields) - {"days_since_last_contact"}
    derived = {"pdays", "previously_contacted"}
    assert caller | derived | set(MARKET_CONTEXT) == set(meta["feature_order"])


def test_the_artifact_explains_itself(meta):
    """A service asked "why did it say that" should not need the training data."""
    reliance = meta["feature_reliance"]
    assert len(reliance) == len(meta["feature_order"])
    assert reliance == sorted(reliance, key=lambda row: -row["drop_in_auc"])
    counts = meta["confusion_at_threshold"]
    assert sum(counts.values()) == meta["n_test"]
