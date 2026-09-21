"""A Pydantic model in the signature becomes the request body.

Nothing registers the schema. The annotation is the registration.
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class CallRequest(BaseModel):
    prospect_id: int
    attempts: int


@app.post("/calls")
def log_call(call: CallRequest):
    return {"prospect_id": call.prospect_id, "attempts": call.attempts}
