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


def test_health_names_the_commit_that_built_the_model(client):
    """Provenance a caller can read without opening the repository."""
    assert client.get("/health").json()["model_commit"]


def test_predict_returns_the_promised_shape(client, prospect):
    body = client.post("/predict", json=prospect).json()
    assert set(body) == {"subscribe_probability", "call", "threshold", "model_version"}
    assert 0.0 <= body["subscribe_probability"] <= 1.0
    assert body["call"] == (body["subscribe_probability"] >= body["threshold"])


def test_the_threshold_served_is_not_the_library_default(client, prospect):
    """If this starts failing at 0.5, somebody replaced the stored threshold."""
    assert client.post("/predict", json=prospect).json()["threshold"] != 0.5


def test_a_typo_in_a_category_is_rejected_not_ignored(client, prospect):
    response = client.post("/predict", json=dict(prospect, month="May"))
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "month"]


def test_economic_context_cannot_be_supplied_by_the_caller(client, prospect):
    response = client.post("/predict", json=dict(prospect, euribor3m=4.9))
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_a_missing_field_is_rejected(client, prospect):
    payload = {k: v for k, v in prospect.items() if k != "poutcome"}
    assert client.post("/predict", json=payload).status_code == 422


def test_never_contacted_is_expressed_as_null_not_999(client, prospect):
    assert client.post("/predict", json=dict(prospect, days_since_last_contact=None)).status_code == 200
    assert client.post("/predict", json=dict(prospect, days_since_last_contact=999)).status_code == 422


def test_batch_matches_one_at_a_time(client, prospect):
    other = dict(prospect, age=58, job="retired", poutcome="success",
                 previous=2, days_since_last_contact=6)
    batch = client.post("/predict/batch", json=[prospect, other]).json()
    singles = [client.post("/predict", json=p).json() for p in (prospect, other)]
    assert batch == singles


def test_the_service_agrees_with_the_pipeline_it_serves(client, prospect):
    """The skew test.

    Score the same prospect twice: once through HTTP, once by calling the
    pipeline directly. A serving layer that reorders a column, derives the
    contact features differently or forgets the economic context will pass
    every other test in this file and fail this one.
    """
    served = client.post("/predict", json=prospect).json()
    bundle = app.state.bundle
    frame = to_frame([Customer(**prospect)], bundle)
    direct = bundle["pipeline"].predict_proba(frame)[0, 1]
    assert served["subscribe_probability"] == pytest.approx(direct, abs=5e-5)
