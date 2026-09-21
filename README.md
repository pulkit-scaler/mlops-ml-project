# Churn API

Companion code for the end to end machine learning project session.

A telecom retention model, served over HTTP. Given one customer, the service
returns the probability that they churn, the decision that probability implies,
and the version of the model that produced it. Nothing else.

The interesting part of this repository is not the model. It is the seam
between the thing that produces the artifact and the thing that consumes it,
and the handful of files that keep those two from drifting apart.

## Layout

```
churn/          produces the artifact      data.py  pipeline.py  train.py
app/            consumes the artifact      schemas.py  main.py
artifacts/      the seam                   churn_pipeline.joblib  model_meta.json
tests/          proves the seam holds      test_data  test_contract  test_api
snippets/       the FastAPI teaching examples
data/           a copy of the Kaggle CSV, for when kagglehub is unavailable
```

`churn/` and `app/` never import each other. They communicate only through
`artifacts/`, which is exactly the boundary the session is about.

## Setup

```bash
git clone https://github.com/pulkit-scaler/mlops-ml-project.git
cd mlops-ml-project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer. Tested on 3.12.

`scikit-learn` is pinned exactly because `churn_pipeline.joblib` was pickled by
that version. The service checks the running version against the one recorded
at training time and reports itself degraded when they differ, rather than
serving predictions from a model unpickled under a version it has never met.

## Run the service

```bash
uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000/docs>, which nobody wrote. It is generated from
the type annotations in `app/schemas.py`.

| Endpoint | What it does |
|---|---|
| `GET /health` | Whether a model is loaded and whether its version matches |
| `POST /predict` | One customer, one probability and one decision |
| `POST /predict/batch` | Up to 1000 customers in one round trip |

A request from the command line:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{"tenure":2,"MonthlyCharges":89.1,"TotalCharges":178.2,"gender":"Female",
       "SeniorCitizen":0,"Partner":"No","Dependents":"No","PhoneService":"Yes",
       "MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No",
       "OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
       "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
       "PaperlessBilling":"Yes","PaymentMethod":"Electronic check"}'
```

## Retrain

```bash
python -m churn.train
```

Refits the pipeline, reselects the decision threshold by cross validation on
the training split, and overwrites both files in `artifacts/`. The threshold is
never chosen on the test set.

## Tests

```bash
pytest
```

Sixteen tests in three groups. `test_data.py` pins the two cleaning decisions,
`test_contract.py` fails if the request schema and the fitted encoder stop
agreeing about which category values exist, and `test_api.py` checks the
service contract, including a test that scores the same customer through HTTP
and through the pipeline directly and requires the two to match.

## The teaching snippets

`snippets/` holds the six short FastAPI scripts from the session, each
demonstrating one idea. Run any of them the same way:

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

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
on Kaggle, originally an IBM sample dataset. 7,043 customers, 19 features,
26.5% churn. `src/churn/data.py` fetches it through `kagglehub`, which needs no
Kaggle account for a public dataset, and falls back to the byte identical copy
under `data/` when there is no network.
