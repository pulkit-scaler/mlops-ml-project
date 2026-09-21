"""Contract tests for the service, including the one that catches skew."""
import pytest
from fastapi.testclient import TestClient

from app.main import app, to_frame
from app.schemas import Customer


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["sklearn_version_matches"] is True


def test_predict_returns_the_promised_shape(client, sample_customer):
    body = client.post("/predict", json=sample_customer).json()
    assert set(body) == {"churn_probability", "churn", "threshold", "model_version"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn"] == (body["churn_probability"] >= body["threshold"])


def test_a_typo_in_a_category_is_rejected_not_ignored(client, sample_customer):
    bad = dict(sample_customer, Contract="Month to month")
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "Contract"]


def test_an_unknown_field_is_rejected(client, sample_customer):
    response = client.post("/predict", json=dict(sample_customer, Loyalty="gold"))
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_a_missing_field_is_rejected(client, sample_customer):
    payload = {k: v for k, v in sample_customer.items() if k != "TechSupport"}
    assert client.post("/predict", json=payload).status_code == 422


def test_a_negative_tenure_is_rejected(client, sample_customer):
    assert client.post("/predict", json=dict(sample_customer, tenure=-1)).status_code == 422


def test_batch_matches_one_at_a_time(client, sample_customer):
    other = dict(sample_customer, Contract="Two year", tenure=60, TotalCharges=4200.0)
    batch = client.post("/predict/batch", json=[sample_customer, other]).json()
    singles = [client.post("/predict", json=p).json() for p in (sample_customer, other)]
    assert batch == singles


def test_the_service_agrees_with_the_pipeline_it_serves(client, sample_customer):
    """The skew test.

    Score the same customer twice: once through HTTP, once by calling the
    pipeline directly the way training does. A serving layer that reorders,
    renames or retypes a column will pass every other test in this file and
    fail this one.
    """
    served = client.post("/predict", json=sample_customer).json()
    bundle = app.state.bundle
    frame = to_frame(Customer(**sample_customer), bundle["meta"])
    direct = bundle["pipeline"].predict_proba(frame)[0, 1]
    assert served["churn_probability"] == pytest.approx(direct, abs=5e-5)
