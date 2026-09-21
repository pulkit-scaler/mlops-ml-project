"""The request schema and the fitted encoder must agree.

This is the test that stops the two halves of the project drifting apart. If
somebody retrains on data with a new PaymentMethod and does not widen the
Literal, this fails before the service accepts a value it will silently ignore.
"""
import typing

from app.schemas import Customer
from churn.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def allowed_values(field):
    return set(typing.get_args(field.annotation))


def test_schema_covers_exactly_the_training_columns(meta):
    assert set(Customer.model_fields) == set(meta["feature_order"])


def test_every_categorical_literal_matches_the_encoder(meta):
    for column in CATEGORICAL_FEATURES:
        schema_values = allowed_values(Customer.model_fields[column])
        encoder_values = set(meta["categories"][column])
        assert schema_values == encoder_values, column


def test_numeric_fields_are_numbers_not_strings():
    for column in NUMERIC_FEATURES:
        assert Customer.model_fields[column].annotation in (int, float), column
