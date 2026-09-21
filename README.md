# Term deposit scoring

An end to end machine learning project, and the companion code for the
end to end session. It puts together the four things the course has built so
far: a repository with a history, a search that chooses the model, a service
that answers over HTTP, and a console a human can use.

A Portuguese bank ran a telephone campaign for term deposits. Given a prospect
and what the campaign knows about them so far, should we place the call.

The interesting part is not the model. It is the seam between the program that
produces the artifact and the programs that consume it, and the handful of
files that keep them from drifting apart.

## Layout

```
campaign/       produces the artifact     data.py  pipeline.py  tune.py  train.py
app/            consumes it over HTTP     schemas.py  main.py
dashboard/      consumes the HTTP service app.py
artifacts/      the seam                  model.joblib  model_meta.json
                                          market_context.json  study.db
tests/          proves the seam holds     data  features  contract  api
snippets/       the FastAPI examples from the session
data/           a compressed copy of the Kaggle CSV, for when there is no network
```

Read the arrows. `campaign/` writes `artifacts/`, `app/` reads them, and
`dashboard/` never touches them at all: it asks the service. There is exactly
one place where a prospect becomes a probability.

`app/` imports one thing from `campaign/`, and that is deliberate. Anything
*fitted* travels inside `model.joblib`. Anything merely *decided*, like the
rule that turns a missing contact date into two columns, has to be shared as
code or it gets implemented twice and eventually differently.

## Setup

```bash
git clone https://github.com/pulkit-scaler/mlops-ml-project.git
cd mlops-ml-project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer. Tested on 3.12. `scikit-learn` is pinned exactly because
`model.joblib` was pickled by that version, and the service checks.

## The four commands

```bash
python -m campaign.tune --trials 60     # search, writes artifacts/study.db
python -m campaign.train                # fit the winner, writes the artifact
uvicorn app.main:app --reload           # serve it at http://127.0.0.1:8000
streamlit run dashboard/app.py          # a console over the service
```

The last two run at the same time, in two terminals. The dashboard is a client
of the service, so the service has to be up first.

Artifacts are committed, so you can skip the first two and go straight to
serving.

## The service

<http://127.0.0.1:8000/docs> is generated from the annotations in
`app/schemas.py`. Nobody wrote it.

| Endpoint | What it does |
|---|---|
| `GET /health` | Whether a model is loaded, and which commit built it |
| `GET /model` | The model card: threshold, metrics, allowed values |
| `POST /predict` | One prospect, one probability and one recommendation |
| `POST /predict/batch` | Up to 1000 prospects in one round trip |

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{"age":49,"campaign":1,"previous":3,"days_since_last_contact":6,
       "job":"admin.","marital":"married","education":"high.school",
       "default":"unknown","housing":"no","loan":"no","contact":"cellular",
       "month":"oct","day_of_week":"wed","poutcome":"success"}'
```

Note what the request does not carry. The five economic columns are not
properties of a prospect, so the service supplies them from
`artifacts/market_context.json`. And there is no `pdays`, because `999` meaning
"never contacted" is a quirk of a CSV and has no place in a public contract.

## The search

`campaign/tune.py` runs an Optuna study whose space branches: the trial picks a
model family first, and only that family's parameters are suggested. The study
is stored in SQLite, so a second run continues the first instead of starting
over, and the trials are still there tomorrow.

```bash
python -m campaign.tune --trials 20     # run it again, it resumes
```

`campaign/train.py` reads the best trial back out and replays it through the
search's own `suggest` function rather than picking the parameter names apart,
so the model that ships is assembled by exactly the code that scored it.

## Provenance

`artifacts/model_meta.json` records the commit that produced the artifact and
whether the working tree was clean at the time. `GET /health` reports it. A
model release also gets an annotated tag:

```bash
git tag -a model-20260921 -m "hgb, test ROC AUC 0.8166" && git push --tags
git show model-20260921
```

That is the cheap end of experiment tracking. It answers "which code produced
this model" and nothing else: it does not record what the previous run scored,
which is the gap the next session fills.

## Tests

```bash
pytest
```

Twenty three tests in four groups. `test_data.py` pins the data decisions,
`test_features.py` checks that the one rule implemented twice means the same
thing both times, `test_contract.py` fails when the request schema and the
fitted encoder stop agreeing, and `test_api.py` checks the service contract
including a test that scores the same prospect through HTTP and through the
pipeline directly and requires the two to match.

## The teaching snippets

`snippets/` holds six short FastAPI scripts, one idea each.

```bash
uvicorn --app-dir snippets 01_hello:app --reload
```

| File | Idea |
|---|---|
| `01_hello.py` | The smallest application, and the docs you did not write |
| `02_types.py` | Type hints are the parser, not documentation |
| `03_body.py` | A Pydantic model in the signature is the request body |
| `04_validation.py` | Constraints belong in the schema |
| `05_response_model.py` | The response is a contract too |
| `06_startup_cost.py` | Where you load the model decides your latency |

## The data

[Bank Marketing](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
on Kaggle, originally from UCI. 41,188 calls, 11.3% subscribed.
`campaign/data.py` fetches it through `kagglehub`, which needs no account for a
public dataset, and falls back to the compressed copy under `data/`.

One column is dropped before anything else. `duration` is how long the call
lasted, it is the strongest predictor in the dataset, and it is unusable: the
model exists to decide whether to make the call. Keeping it takes test ROC AUC
from 0.80 to 0.95 and produces a model that cannot be deployed, because the
serving code has nothing to put in the field. The dataset's own documentation
says to discard it. Most published notebooks on this data do not.
