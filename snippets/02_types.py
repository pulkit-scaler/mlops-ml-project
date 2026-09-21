"""Type hints are not documentation here. They are the parser.

customer_id is declared int, so /customers/abc never reaches the function:
the framework rejects it with a 422 and a message naming the field.
verbose is declared bool, so "yes", "true", "1" and "on" all arrive as True.
"""
from fastapi import FastAPI

app = FastAPI()


@app.get("/customers/{customer_id}")
def get_customer(customer_id: int, verbose: bool = False):
    body = {"customer_id": customer_id, "type": type(customer_id).__name__}
    if verbose:
        body["note"] = "the path segment was a string in the URL and an int in here"
    return body
