"""Shared fixtures. Tests read the committed CSV so they never touch the network."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from churn.data import LOCAL_CSV, clean  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return pd.read_csv(LOCAL_CSV)


@pytest.fixture(scope="session")
def cleaned(raw):
    return clean(raw)


@pytest.fixture(scope="session")
def meta():
    return json.loads((ROOT / "artifacts" / "model_meta.json").read_text())


@pytest.fixture(scope="session")
def sample_customer():
    """A short tenure, month to month, fiber customer. The risky profile."""
    return {
        "tenure": 2,
        "MonthlyCharges": 89.1,
        "TotalCharges": 178.2,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }
