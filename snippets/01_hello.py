"""The smallest FastAPI application.

    uvicorn --app-dir snippets 01_hello:app --reload

Then open http://127.0.0.1:8000/ping and http://127.0.0.1:8000/docs.
The second one you did not write. It is generated from the first.
"""
from fastapi import FastAPI

app = FastAPI()


@app.get("/ping")
def ping():
    return {"pong": True}
