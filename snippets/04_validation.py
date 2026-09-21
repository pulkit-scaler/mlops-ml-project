"""Constraints belong in the schema, not in the first four lines of the handler.

Every rule below is enforced before create_order runs, reported as a 422 with
the offending field named, and published in the OpenAPI document so a caller
can see the rule without reading this file.
"""
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI()


class Order(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: str = Field(min_length=3, max_length=12)
    quantity: int = Field(ge=1, le=500)
    channel: Literal["web", "store", "phone"]


@app.post("/orders")
def create_order(order: Order):
    return {"accepted": order.model_dump()}
