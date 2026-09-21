"""The response is a contract too.

internal_score is computed and returned by the function, and the caller never
sees it, because response_model decides what leaves the process. Adding a field
to the handler does not silently add it to the public API.
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Quote(BaseModel):
    price: float
    currency: str


@app.get("/quote", response_model=Quote)
def quote():
    return {"price": 19.99, "currency": "GBP", "internal_score": 0.83}
