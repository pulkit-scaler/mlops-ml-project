"""Constraints belong in the schema, not in the first four lines of the handler.

Every rule below is enforced before log_call runs, reported as a 422 with the
offending field named, and published in the OpenAPI document so a caller can
see the rule without reading this file.
"""
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI()


class CallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prospect_id: int = Field(ge=1)
    attempts: int = Field(ge=1, le=50)
    channel: Literal["cellular", "telephone"]


@app.post("/calls")
def log_call(call: CallRequest):
    return {"accepted": call.model_dump()}
