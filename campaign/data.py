"""Load the bank marketing table and put it in the shape the pipeline expects.

Kaggle is the source of record. load_raw reads a kagglehub download when there
is a network and otherwise reads the compressed copy under data/, which is the
same file. It returns the frame and a one line description of where the frame
came from, because a loader that silently changes source is a loader you cannot
debug.

Three decisions live in this module and nowhere else, so that training and
serving cannot drift apart. Each one is explained at the point it is made.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CSV = REPO_ROOT / "data" / "bank-additional-full.csv.gz"
KAGGLE_DATASET = "henriqueyamahata/bank-marketing"
KAGGLE_FILE = "bank-additional-full.csv"

TARGET = "y"

# Known only after the call has happened, so it cannot be in a request that
# decides whether to make the call. See drop_leakage below.
LEAKING = "duration"

# What the caller knows about the customer and the campaign so far.
CUSTOMER_NUMERIC = ["age", "campaign", "pdays", "previous", "previously_contacted"]

# What a request actually carries. pdays and previously_contacted are derived
# from days_since_last_contact by contact_features, because 999 is a quirk of
# this file and has no business in a public contract.
REQUEST_NUMERIC = ["age", "campaign", "previous", "days_since_last_contact"]
CUSTOMER_CATEGORICAL = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

# Not properties of the customer at all. These describe the economy on the day
# of the call, they are identical for every customer contacted that day, and a
# caller has no business supplying them. The service fills them in. See
# artifacts/market_context.json and app/main.py.
MARKET_CONTEXT = [
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

NUMERIC_FEATURES = CUSTOMER_NUMERIC + MARKET_CONTEXT
CATEGORICAL_FEATURES = CUSTOMER_CATEGORICAL
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw():
    """Return the untouched Kaggle table and where it was read from."""
    try:
        import kagglehub

        folder = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        csv = next(folder.rglob(KAGGLE_FILE))
        return pd.read_csv(csv, sep=";"), f"kagglehub -> {csv.name}"
    except Exception as exc:
        frame = pd.read_csv(LOCAL_CSV, sep=";")
        return frame, f"repo copy -> {LOCAL_CSV.name} (kagglehub raised {type(exc).__name__})"


def drop_leakage(df):
    """Remove the one column that cannot exist when the prediction is needed.

    duration is the length of the call in seconds. It is the strongest single
    predictor in this dataset and it is unusable, because the model's whole job
    is to decide whether to place the call. Keeping it produces a model that
    scores beautifully in a notebook and cannot be deployed, since the serving
    code has nothing to put in the field.

    The dataset's own documentation says to discard it. Most published
    notebooks on this data do not.
    """
    return df.drop(columns=[LEAKING])


NEVER_CONTACTED = 999


def contact_features(days_since_last_contact):
    """One customer's contact history to the two columns the model expects.

    None means never contacted. This is the serving side of the same rule
    encode_pdays applies to a whole frame, and the two are checked against
    each other in tests/test_features.py.

    It lives here, in the package that trains, rather than being retyped in
    the service. Fitted transformations travel inside the artifact. A
    derivation like this one is not fitted, it is decided, so it has to travel
    as shared code or it will be implemented twice and eventually differently.
    """
    never = days_since_last_contact is None
    return {
        "pdays": float("nan") if never else float(days_since_last_contact),
        "previously_contacted": 0 if never else 1,
    }


def encode_pdays(df):
    """Turn the 999 sentinel into a value and a flag.

    pdays is days since the customer was last contacted, and 999 means never.
    For 96 per cent of rows it is 999. Left alone, the model is told that
    almost everybody was contacted about three years ago, which is not a
    slightly wrong number, it is a category wearing a number's clothes.

    Splitting it costs one extra column: previously_contacted says whether the
    question applies, and pdays answers it only when it does.
    """
    out = df.copy()
    out["previously_contacted"] = (out["pdays"] != NEVER_CONTACTED).astype(int)
    out["pdays"] = out["pdays"].replace(NEVER_CONTACTED, np.nan)
    return out


def clean(raw):
    """Return (X, y) ready for the pipeline."""
    df = raw.copy()
    y = (df.pop(TARGET).astype(str).str.strip() == "yes").astype(int)
    df = encode_pdays(drop_leakage(df))
    return df[FEATURE_ORDER], y


def latest_market_context(raw):
    """The most recent economic readings in the training data.

    The service needs values for the five context columns and the caller cannot
    supply them. Until something queries a real feed, the honest stand in is
    the last window this model was trained on, recorded with the artifact so
    that a stale context is visible rather than assumed.
    """
    order = ["nr.employed", "euribor3m"]
    last = raw.sort_values(order).iloc[-1]
    return {column: float(last[column]) for column in MARKET_CONTEXT}
