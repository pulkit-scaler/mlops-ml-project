"""The term deposit scoring service.

    uvicorn app.main:app --reload

Four decisions in here are the reason this file is worth reading rather than
copying, and each has a lazier version that works on a laptop and hurts later.

The pipeline is loaded once, in the lifespan handler, not inside the endpoint.

The feature frame is built from meta["feature_order"], not a list typed here.

The decision uses the threshold stored with the model. Calling predict() would
silently mean 0.5, and 0.5 is not what this model was tuned for.

The economic context is read from the artifact, not from the request. Callers
do not know the euribor rate and should not be asked to.
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request

from campaign.data import contact_features

from .schemas import Customer, Health, Prediction

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS / "model.joblib"
META_PATH = ARTIFACTS / "model_meta.json"
CONTEXT_PATH = ARTIFACTS / "market_context.json"


def load_bundle():
    """Read the artifact and its metadata. Raises if any of the three is absent."""
    return {
        "pipeline": joblib.load(MODEL_PATH),
        "meta": json.loads(META_PATH.read_text()),
        "context": json.loads(CONTEXT_PATH.read_text()),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup. A failure here stops the process, which is what you
    # want: a service that starts without a model looks healthy to whatever is
    # watching it and returns errors to real traffic.
    try:
        app.state.bundle = load_bundle()
    except Exception:
        app.state.bundle = None
        raise
    yield
    app.state.bundle = None


app = FastAPI(
    title="Term deposit scoring",
    description="Should this prospect be called about a term deposit.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_bundle(request: Request):
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return bundle


def to_row(customer: Customer, bundle) -> dict:
    """One validated request to one row of model inputs.

    Three sources meet here: the fields the caller sent, the two contact
    columns derived by the shared rule, and the economic context the service
    holds. Nothing in this function knows a column name that is not in the
    metadata.
    """
    row = customer.model_dump()
    row.update(contact_features(row.pop("days_since_last_contact")))
    row.update(bundle["context"])
    return row


def to_frame(customers, bundle) -> pd.DataFrame:
    rows = [to_row(c, bundle) for c in customers]
    return pd.DataFrame(rows)[bundle["meta"]["feature_order"]]


def score(frame: pd.DataFrame, bundle) -> list[Prediction]:
    meta = bundle["meta"]
    threshold = meta["decision_threshold"]
    proba = bundle["pipeline"].predict_proba(frame)[:, 1]
    return [
        Prediction(
            subscribe_probability=round(float(p), 4),
            call=bool(p >= threshold),
            threshold=threshold,
            model_version=meta["model_version"],
        )
        for p in proba
    ]


@app.get("/health", response_model=Health)
def health(request: Request):
    """Whether this process can serve a prediction right now.

    It reports on the model rather than on the web server. A health check that
    only proves the process is listening will pass happily while every
    prediction fails.
    """
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return Health(
            status="degraded",
            model_loaded=False,
            model_version=None,
            model_commit=None,
            sklearn_version_matches=False,
        )
    meta = bundle["meta"]
    matches = meta["sklearn_version"] == sklearn.__version__
    return Health(
        status="ok" if matches else "degraded",
        model_loaded=True,
        model_version=meta["model_version"],
        model_commit=(meta.get("git") or {}).get("commit"),
        sklearn_version_matches=matches,
    )


@app.get("/model")
def model_card(request: Request):
    """What the dashboard and any other client need to render themselves.

    Publishing the metadata means a client does not hardcode the threshold,
    the allowed values or the feature list. It asks.
    """
    meta = get_bundle(request)["meta"]
    return {
        "model_version": meta["model_version"],
        "family": meta["family"],
        "trained_at": meta["trained_at"],
        "git": meta["git"],
        "decision_threshold": meta["decision_threshold"],
        "metrics": meta["metrics"],
        "study": meta["study"],
        "categories": meta["categories"],
        "confusion_at_threshold": meta["confusion_at_threshold"],
        "feature_reliance": meta["feature_reliance"],
    }


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, request: Request):
    bundle = get_bundle(request)
    return score(to_frame([customer], bundle), bundle)[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer], request: Request):
    """Many prospects, one HTTP round trip and one vectorised predict_proba."""
    if not customers:
        raise HTTPException(status_code=422, detail="send at least one customer")
    if len(customers) > 1000:
        raise HTTPException(status_code=413, detail="at most 1000 customers per call")
    bundle = get_bundle(request)
    return score(to_frame(customers, bundle), bundle)
