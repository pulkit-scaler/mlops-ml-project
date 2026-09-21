"""Type hints are not documentation here. They are the parser.

prospect_id is declared int, so /prospects/abc never reaches the function:
the framework rejects it with a 422 and a message naming the field.
verbose is declared bool, so "yes", "true", "1" and "on" all arrive as True.
"""
from fastapi import FastAPI

app = FastAPI()


@app.get("/prospects/{prospect_id}")
def get_prospect(prospect_id: int, verbose: bool = False):
    body = {"prospect_id": prospect_id, "type": type(prospect_id).__name__}
    if verbose:
        body["note"] = "the path segment was a string in the URL and an int in here"
    return body
