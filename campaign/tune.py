"""Choose the model with Optuna.

    python -m campaign.tune --trials 60

Two things here are the reason Optuna is in this project rather than a grid.

The space branches. Which parameters exist depends on which family the trial
picked, and there is no rectangle that expresses that. suggest_categorical
chooses the family and the following suggest_ calls are only reached for that
branch, which is what define-by-run means.

The study outlives the process. It is stored in a SQLite file, so a second run
with the same study name continues the first rather than starting over, and the
trials are still there tomorrow when somebody asks what was tried.
"""
import argparse
import sys
from pathlib import Path

import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .data import clean, load_raw
from .pipeline import build_pipeline

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
STORAGE = f"sqlite:///{ARTIFACTS / 'study.db'}"
STUDY_NAME = "term-deposit"
RANDOM_STATE = 42
SCORING = "roc_auc"


def suggest(trial):
    """One trial to a (family, params) pair. The space branches here."""
    family = trial.suggest_categorical("family", ["logreg", "hgb", "rf"])

    if family == "logreg":
        return family, {
            "C": trial.suggest_float("lr_C", 1e-3, 1e2, log=True),
            "class_weight": trial.suggest_categorical("lr_class_weight", [None, "balanced"]),
        }
    if family == "hgb":
        return family, {
            "learning_rate": trial.suggest_float("hgb_learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("hgb_max_leaf_nodes", 4, 64),
            "min_samples_leaf": trial.suggest_int("hgb_min_samples_leaf", 5, 150),
            "l2_regularization": trial.suggest_float("hgb_l2", 1e-4, 10.0, log=True),
            "max_iter": trial.suggest_int("hgb_max_iter", 60, 400),
        }
    return family, {
        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 500),
        "max_depth": trial.suggest_int("rf_max_depth", 3, 24),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 40),
        "class_weight": trial.suggest_categorical("rf_class_weight", [None, "balanced"]),
    }


def make_objective(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        family, params = suggest(trial)
        pipeline = build_pipeline(family, params)
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1
        )
        return scores.mean()

    return objective


def get_study():
    """Open the stored study, or create it. load_if_exists makes reruns resume."""
    ARTIFACTS.mkdir(exist_ok=True)
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        load_if_exists=True,
    )


def run(n_trials, X_train, y_train):
    study = get_study()
    already = len(study.trials)
    study.optimize(make_objective(X_train, y_train), n_trials=n_trials)
    return study, already


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=60)
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.model_selection import train_test_split

    raw, source = load_raw()
    X, y = clean(raw)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    study, already = run(args.trials, X_train, y_train)
    best = study.best_trial

    print(f"data          : {source}")
    print(f"study         : {STUDY_NAME} in {ARTIFACTS / 'study.db'}")
    print(f"trials        : {already} before this run, {len(study.trials)} now")
    print(f"best {SCORING:9}: {study.best_value:.4f}")
    print(f"best family   : {best.params['family']}")
    for key, value in best.params.items():
        if key != "family":
            print(f"    {key:24} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
