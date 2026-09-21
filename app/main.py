"""The churn service.

    uvicorn app.main:app --reload

Three things in here are deliberate and are the reason this file is worth
reading rather than copying.

The pipeline is loaded once, in the lifespan handler, before the first request
is served. Loading it inside the endpoint would add the read and the unpickle
to every single call.

The feature frame is built from meta["feature_order"], not from a literal list
typed into this file. A list typed here is a list that goes stale the next time
someone adds a column to the training data.

The decision comes from the threshold stored alongside the model, not from
predict(). calling predict() silently means 0.5, and 0.5 is not the number this
model was tuned for.
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request

from .schemas import Customer, Health, Prediction

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS / "churn_pipeline.joblib"
META_PATH = ARTIFACTS / "model_meta.json"


def load_bundle():
    """Read the artifact and its metadata. Raises if either is absent."""
    meta = json.loads(META_PATH.read_text())
    return {"pipeline": joblib.load(MODEL_PATH), "meta": meta}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup. A failure here stops the process, which is what you
    # want: a service that starts without a model is a service that returns
    # 500s to real traffic instead of failing in the deploy.
    try:
        app.state.bundle = load_bundle()
    except Exception:
        app.state.bundle = None
        raise
    yield
    app.state.bundle = None


app = FastAPI(
    title="Churn API",
    description="Churn probability for a single telecom customer.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_bundle(request: Request):
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return bundle


def to_frame(customer: Customer, meta) -> pd.DataFrame:
    """One validated customer to the one row frame the pipeline expects."""
    return pd.DataFrame([customer.model_dump()])[meta["feature_order"]]


def score(frame: pd.DataFrame, bundle) -> list[Prediction]:
    meta = bundle["meta"]
    threshold = meta["decision_threshold"]
    proba = bundle["pipeline"].predict_proba(frame)[:, 1]
    return [
        Prediction(
            churn_probability=round(float(p), 4),
            churn=bool(p >= threshold),
            threshold=threshold,
            model_version=meta["model_version"],
        )
        for p in proba
    ]


@app.get("/health", response_model=Health)
def health(request: Request):
    """Is this process able to serve a prediction right now.

    It reports on the model rather than on the web server. A health check that
    only proves the process is listening will happily pass while every
    prediction fails.
    """
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return Health(
            status="degraded",
            model_loaded=False,
            model_version=None,
            sklearn_version_matches=False,
        )
    meta = bundle["meta"]
    matches = meta["sklearn_version"] == sklearn.__version__
    return Health(
        status="ok" if matches else "degraded",
        model_loaded=True,
        model_version=meta["model_version"],
        sklearn_version_matches=matches,
    )


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, request: Request):
    bundle = get_bundle(request)
    return score(to_frame(customer, bundle["meta"]), bundle)[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer], request: Request):
    """Many customers, one HTTP round trip and one vectorised predict_proba."""
    if not customers:
        raise HTTPException(status_code=422, detail="send at least one customer")
    if len(customers) > 1000:
        raise HTTPException(status_code=413, detail="at most 1000 customers per call")
    bundle = get_bundle(request)
    frame = pd.DataFrame([c.model_dump() for c in customers])[bundle["meta"]["feature_order"]]
    return score(frame, bundle)
