"""Load the Telco churn table and put it in the shape the pipeline expects.

Kaggle is the source of record. load_raw reads a kagglehub download when Kaggle
credentials are configured and otherwise reads the copy committed under data/,
which is the same file. It returns the frame and a one line description of
where the frame came from, because a loader that silently changes source is a
loader you cannot debug.
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CSV = REPO_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
KAGGLE_DATASET = "blastchar/telco-customer-churn"

TARGET = "Churn"
IDENTIFIER = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw():
    """Return the untouched Kaggle table and where it was read from."""
    try:
        import kagglehub

        folder = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        csv = next(folder.glob("*.csv"))
        return pd.read_csv(csv), f"kagglehub -> {csv.name}"
    except Exception as exc:
        reason = f"{type(exc).__name__}"
        return pd.read_csv(LOCAL_CSV), f"repo copy -> {LOCAL_CSV.name} (kagglehub raised {reason})"


def clean(raw):
    """Return (X, y) ready for the pipeline.

    Two decisions live here and nowhere else, so that training and serving
    cannot drift apart.

    TotalCharges arrives as text. Eleven rows hold a single space instead of a
    number, which is enough to make pandas type the whole column as object.
    Every one of those eleven has tenure 0: they are customers who signed up
    and have not been billed yet. Their charge to date is genuinely zero, so
    that is what we write. A median would invent a billing history.

    customerID is dropped. It identifies the row rather than describing the
    customer, and a model given a unique key will happily use it.
    """
    df = raw.copy()
    df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    y = df.pop(TARGET)
    X = df.drop(columns=[IDENTIFIER])[FEATURE_ORDER]
    return X, y
