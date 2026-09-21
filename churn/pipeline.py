"""The preprocessing and model pipeline, as one estimator.

The whole point of building this as a single Pipeline is that it is the only
object that carries the training time transformations with it. A fitted
ColumnTransformer knows which categories it saw and what the numeric means and
scales were. Serving code that rebuilds an encoder from the request cannot
know any of that, which is how train and serve drift apart.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_pipeline(C=1.0, class_weight=None, random_state=42):
    """One estimator: raw customer columns in, churn probability out."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    # handle_unknown="ignore" decides what a value the encoder never saw does
    # at serving time. See the failure modes section of the class: this setting
    # trades a loud crash for a quiet all zeros row, and neither is free.
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
            (
                "model",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )
