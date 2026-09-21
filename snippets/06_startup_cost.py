"""Where you load the model decides your latency.

/slow reads and unpickles the artifact inside the handler, so every caller
pays for it. /fast uses the object loaded once in the lifespan handler. Same
model, same answer, and the session measures the difference.
"""
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = joblib.load(MODEL_PATH)
    yield
    app.state.pipeline = None


app = FastAPI(lifespan=lifespan)


@app.get("/slow")
def slow():
    pipeline = joblib.load(MODEL_PATH)
    return {"n_features": len(pipeline.named_steps["preprocess"].feature_names_in_)}


@app.get("/fast")
def fast():
    pipeline = app.state.pipeline
    return {"n_features": len(pipeline.named_steps["preprocess"].feature_names_in_)}
