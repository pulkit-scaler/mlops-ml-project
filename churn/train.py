"""Fit the churn pipeline and write the two files the service loads.

    python -m churn.train

Everything the serving layer needs to know about this model goes into
artifacts/model_meta.json. The pipeline knows how to transform and predict. It
does not know which threshold the business agreed to, which categories the
request schema should accept, or which version of itself is running. A service
that has to guess any of those is a service that will one day guess wrong.
"""
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .data import (
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    clean,
    load_raw,
)
from .pipeline import build_pipeline

RANDOM_STATE = 42
TEST_SIZE = 0.25
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS / "churn_pipeline.joblib"
META_PATH = ARTIFACTS / "model_meta.json"


def pick_threshold(pipe, X, y):
    """Choose the decision threshold on the training set, never on the test set.

    Cross validated probabilities give every training row a prediction from a
    model that did not see it, which is the only honest way to tune a threshold
    without spending the holdout.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.20, 0.81, 0.01), 2)
    scores = [f1_score(y, (proba >= t).astype(int)) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def main():
    raw, source = load_raw()
    X, y = clean(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipe = build_pipeline()
    threshold, cv_f1 = pick_threshold(pipe, X_train, y_train)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    at_half = (proba >= 0.5).astype(int)
    at_chosen = (proba >= threshold).astype(int)

    encoder = pipe.named_steps["preprocess"].named_transformers_["cat"]
    categories = {
        col: [c.item() if hasattr(c, "item") else c for c in values]
        for col, values in zip(CATEGORICAL_FEATURES, encoder.categories_)
    }

    meta = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d") + "-logreg-1",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": source,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_order": FEATURE_ORDER,
        "categories": categories,
        "decision_threshold": threshold,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_positive_rate": round(float(y_train.mean()), 4),
        "metrics": {
            "cv_f1_at_threshold": round(cv_f1, 4),
            "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "test_f1_at_half": round(float(f1_score(y_test, at_half)), 4),
            "test_f1_at_threshold": round(float(f1_score(y_test, at_chosen)), 4),
            "test_precision_at_threshold": round(float(precision_score(y_test, at_chosen)), 4),
            "test_recall_at_threshold": round(float(recall_score(y_test, at_chosen)), 4),
        },
    }

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"data              : {source}")
    print(f"train / test      : {meta['n_train']} / {meta['n_test']}")
    print(f"threshold         : {threshold:.2f}  (cross validated F1 {cv_f1:.4f})")
    print(f"test ROC AUC      : {meta['metrics']['test_roc_auc']:.4f}")
    print(f"test F1 at 0.50   : {meta['metrics']['test_f1_at_half']:.4f}")
    print(f"test F1 at {threshold:.2f}   : {meta['metrics']['test_f1_at_threshold']:.4f}")
    print(f"wrote {MODEL_PATH.name} and {META_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
