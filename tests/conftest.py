"""Shared fixtures. Tests read the committed copy so they never need a network."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campaign.data import LOCAL_CSV, clean  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return pd.read_csv(LOCAL_CSV, sep=";")


@pytest.fixture(scope="session")
def cleaned(raw):
    return clean(raw)


@pytest.fixture(scope="session")
def meta():
    return json.loads((ROOT / "artifacts" / "model_meta.json").read_text())


@pytest.fixture(scope="session")
def prospect():
    """A cold prospect: never contacted, no previous campaign."""
    return {
        "age": 41, "campaign": 1, "previous": 0, "days_since_last_contact": None,
        "job": "admin.", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "day_of_week": "mon", "poutcome": "nonexistent",
    }
