"""A Pydantic model in the signature becomes the request body.

Nothing registers the schema. The annotation is the registration.
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Order(BaseModel):
    sku: str
    quantity: int


@app.post("/orders")
def create_order(order: Order):
    return {"sku": order.sku, "units": order.quantity, "total_is_known": False}
