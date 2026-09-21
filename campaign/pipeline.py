"""Preprocessing and the model, as one estimator.

A fitted Pipeline is the only object that carries the training time
transformations with it. It knows which categories the encoder saw and what
the numeric medians and scales were. Serving code that rebuilds an encoder
from a single request cannot know any of that, which is how train and serve
drift apart.

build_pipeline takes a family name and its parameters rather than a model
object, because the Optuna study chooses the family. Keeping that choice in
one function means the search and the final fit cannot disagree about how a
model is assembled.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FAMILIES = ("logreg", "hgb", "rf")
RANDOM_STATE = 42


def build_model(family, params):
    """One family name and a parameter dict to an unfitted estimator."""
    if family == "logreg":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **params)
    if family == "hgb":
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE, **params)
    if family == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    raise ValueError(f"unknown family {family!r}, expected one of {FAMILIES}")


def build_pipeline(family="hgb", params=None):
    """Raw campaign columns in, subscription probability out."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    # handle_unknown="ignore" decides what a value the encoder never saw does at
    # serving time. It trades a loud crash for a quiet row of zeros, and the
    # request schema is where that trade is actually settled.
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocess = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", build_model(family, params or {})),
        ]
    )
