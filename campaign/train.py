"""Fit the chosen model and write the files the service loads.

    python -m campaign.train

This reads the best trial out of the stored Optuna study rather than searching
again, so tuning and training are separate commands with separate costs. Tuning
is slow and occasional. Training is quick and happens whenever the data or the
code changes.

Everything the serving layer needs to know goes into artifacts/model_meta.json.
The pipeline knows how to transform and predict. It does not know which
threshold the business agreed to, which category values the request schema
should accept, which economy it was trained against, or which commit produced
it. A service that has to guess any of those will one day guess wrong.
"""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .data import (
    CATEGORICAL_FEATURES,
    CUSTOMER_CATEGORICAL,
    CUSTOMER_NUMERIC,
    FEATURE_ORDER,
    MARKET_CONTEXT,
    NUMERIC_FEATURES,
    clean,
    latest_market_context,
    load_raw,
)
from .pipeline import build_pipeline
from .tune import get_study, suggest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.joblib"
META_PATH = ARTIFACTS / "model_meta.json"
CONTEXT_PATH = ARTIFACTS / "market_context.json"

RANDOM_STATE = 42
TEST_SIZE = 0.25


def git_provenance():
    """Which commit produced this artifact, and was the tree clean.

    This is the cheapest useful provenance there is. Without it, "the model
    scored 0.80" is a claim nobody can reproduce, because the code that
    produced it has already moved on. dirty=True means the working tree had
    uncommitted changes, so the commit alone does not describe the run.
    """
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
        )

    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"commit": None, "dirty": None, "note": "not a git repository"}
    status = git("status", "--porcelain")
    return {
        "commit": head.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
        "note": None,
    }


def best_configuration():
    """The winning family and parameters from the stored study.

    The best trial is replayed through the search's own suggest function using
    a FixedTrial, rather than by picking the parameter names apart here. That
    matters: a trial records "hgb_l2", the estimator wants "l2_regularization",
    and any translation written in this file is a second place the mapping can
    be wrong. Replaying it means the model trained is assembled by exactly the
    code that scored it.
    """
    study = get_study()
    if not study.trials:
        raise SystemExit(
            "the study is empty. Run 'python -m campaign.tune --trials 60' first."
        )
    best = study.best_trial
    family, params = suggest(optuna.trial.FixedTrial(best.params))
    return family, params, float(study.best_value), len(study.trials)


def pick_threshold(pipeline, X, y):
    """Choose the decision threshold on the training set, never on the test set.

    Cross validated probabilities give every training row a prediction from a
    model that did not see it, which is the only honest way to tune a threshold
    without spending the holdout.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.05, 0.71, 0.01), 2)
    scores = [f1_score(y, (proba >= t).astype(int)) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def reliance(pipeline, X, y, sample=3000):
    """How much the fitted model leans on each input column.

    Permutation importance shuffles one column at a time and measures what
    the score loses. It is computed on held out data and it is a description,
    not a selection step: nothing here changes the model.

    It goes in the metadata because the question "why did it say that" gets
    asked of a service, not of a notebook, and the answer should not require
    anybody to have the training data to hand.
    """
    if len(X) > sample:
        X = X.sample(sample, random_state=RANDOM_STATE)
        y = y.loc[X.index]
    result = permutation_importance(
        pipeline, X, y, scoring="roc_auc", n_repeats=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    ranked = sorted(
        zip(X.columns, result.importances_mean, result.importances_std),
        key=lambda row: -row[1],
    )
    return [
        {"feature": name, "drop_in_auc": round(float(mean), 4), "sd": round(float(sd), 4)}
        for name, mean, sd in ranked
    ]


def main():
    raw, source = load_raw()
    X, y = clean(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    family, params, study_score, n_trials = best_configuration()
    pipeline = build_pipeline(family, params)
    threshold, cv_f1 = pick_threshold(pipeline, X_train, y_train)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    at_half = (proba >= 0.5).astype(int)
    at_chosen = (proba >= threshold).astype(int)

    encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    categories = {
        column: [v.item() if hasattr(v, "item") else v for v in values]
        for column, values in zip(CATEGORICAL_FEATURES, encoder.categories_)
    }
    context = latest_market_context(raw)
    tn, fp, fn, tp = confusion_matrix(y_test, at_chosen).ravel()
    importances = reliance(pipeline, X_test, y_test)

    meta = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d") + f"-{family}-1",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_provenance(),
        "data_source": source,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "family": family,
        "params": params,
        "study": {"trials": n_trials, "best_cv_roc_auc": round(study_score, 4)},
        "customer_numeric": CUSTOMER_NUMERIC,
        "customer_categorical": CUSTOMER_CATEGORICAL,
        "market_context": MARKET_CONTEXT,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_order": FEATURE_ORDER,
        "categories": categories,
        "decision_threshold": threshold,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_positive_rate": round(float(y_train.mean()), 4),
        "confusion_at_threshold": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
        "feature_reliance": importances,
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
    joblib.dump(pipeline, MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    CONTEXT_PATH.write_text(json.dumps(context, indent=2) + "\n")

    provenance = meta["git"]["commit"]
    short = provenance[:8] if provenance else "no commit"
    print(f"data            : {source}")
    print(f"study           : {n_trials} trials, best cross validated ROC AUC {study_score:.4f}")
    print(f"chosen family   : {family}")
    print(f"train / test    : {meta['n_train']:,} / {meta['n_test']:,}")
    print(f"threshold       : {threshold:.2f}  (cross validated F1 {cv_f1:.4f})")
    print(f"test ROC AUC    : {meta['metrics']['test_roc_auc']:.4f}")
    print(f"test F1 at 0.50 : {meta['metrics']['test_f1_at_half']:.4f}")
    print(f"test F1 at {threshold:.2f} : {meta['metrics']['test_f1_at_threshold']:.4f}")
    print(f"calls made      : {tp + fp:,}, wasted {fp:,} ({fp / (tp + fp):.0%})")
    print(f"subscribers      : reached {tp:,}, missed {fn:,}")
    print(f"leans most on   : {', '.join(i['feature'] for i in importances[:3])}")
    print(f"built from      : {short}{' (dirty tree)' if meta['git']['dirty'] else ''}")
    print(f"wrote {MODEL_PATH.name}, {META_PATH.name} and {CONTEXT_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
