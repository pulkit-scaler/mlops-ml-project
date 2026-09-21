"""A console over the scoring service.

    streamlit run dashboard/app.py

This app holds no model. It has no joblib file, no scikit-learn import and no
threshold of its own. Everything it knows it asks the service for, which is the
point: there is exactly one place where a prospect becomes a probability, and
this is not it.

The alternative, loading model.joblib here, is one import shorter and gives you
two implementations of scoring to keep in step forever. The dropdown options
below come from the service's /model endpoint for the same reason. Hardcoding
them would mean this file silently disagreeing with the encoder the first time
somebody retrains.
"""
import pandas as pd
import requests
import streamlit as st

DEFAULT_API = "http://127.0.0.1:8000"
TIMEOUT = 10

st.set_page_config(page_title="Term deposit console", layout="wide")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_model_card(api):
    """Cached for a minute. Every widget below depends on it, and a Streamlit
    script reruns top to bottom on every interaction, so without the cache
    this would be one HTTP call per keystroke."""
    response = requests.get(f"{api}/model", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_health(api):
    return requests.get(f"{api}/health", timeout=TIMEOUT).json()


def post(api, path, payload):
    response = requests.post(f"{api}{path}", json=payload, timeout=TIMEOUT)
    return response.status_code, response.json()


def to_records(frame):
    """A DataFrame to JSON safe rows.

    Two conversions that bite every time. A column holding a missing number is
    float64, so the gap is NaN, and NaN is not valid JSON: it has to become
    null. And the values are numpy scalars, which the json module does not
    know how to encode. astype(object) makes room for None, and .item()
    unwraps numpy to plain Python.
    """
    wide = frame.astype(object).where(pd.notna(frame), None)
    return [
        {key: (value.item() if hasattr(value, "item") else value)
         for key, value in row.items()}
        for row in wide.to_dict("records")
    ]


st.title("Term deposit console")

api = st.sidebar.text_input("Service address", DEFAULT_API)

try:
    health = fetch_health(api)
    card = fetch_model_card(api)
except requests.exceptions.RequestException as exc:
    # A console whose backend is down should say so in a sentence. The
    # alternative is a stack trace in front of whoever is using it.
    st.error(f"No answer from {api}. Start the service with `uvicorn app.main:app`.")
    st.caption(f"{type(exc).__name__}: {exc}")
    st.stop()

status = health["status"]
st.sidebar.metric("Service", status)
st.sidebar.write(f"**Model** `{card['model_version']}`")
st.sidebar.write(f"**Family** {card['family']}")
commit = (card.get("git") or {}).get("commit")
st.sidebar.write(f"**Commit** `{commit[:8] if commit else 'unknown'}`")
st.sidebar.write(f"**Threshold** {card['decision_threshold']}")
if status != "ok":
    st.sidebar.warning("Service reports degraded. Check /health.")

categories = card["categories"]
threshold = card["decision_threshold"]

one, many, about = st.tabs(["Score a prospect", "Score a list", "About this model"])

with one:
    left, middle, right = st.columns(3)
    with left:
        age = st.number_input("Age", 17, 110, 41)
        job = st.selectbox("Job", categories["job"])
        marital = st.selectbox("Marital status", categories["marital"])
        education = st.selectbox("Education", categories["education"], index=6)
    with middle:
        default = st.selectbox("Credit in default", categories["default"])
        housing = st.selectbox("Housing loan", categories["housing"])
        loan = st.selectbox("Personal loan", categories["loan"])
        contact = st.selectbox("Contact type", categories["contact"])
    with right:
        month = st.selectbox("Month", categories["month"])
        day_of_week = st.selectbox("Day of week", categories["day_of_week"])
        campaign = st.number_input("Contacts this campaign", 1, 100, 1)
        previous = st.number_input("Contacts before this campaign", 0, 100, 0)

    poutcome = st.selectbox("Outcome of the previous campaign", categories["poutcome"])
    contacted_before = st.checkbox("Contacted in an earlier campaign", value=False)
    days_since = (
        st.slider("Days since that contact", 0, 30, 6) if contacted_before else None
    )

    payload = {
        "age": int(age), "campaign": int(campaign), "previous": int(previous),
        "days_since_last_contact": days_since,
        "job": job, "marital": marital, "education": education, "default": default,
        "housing": housing, "loan": loan, "contact": contact, "month": month,
        "day_of_week": day_of_week, "poutcome": poutcome,
    }

    if st.button("Score", type="primary"):
        code, body = post(api, "/predict", payload)
        if code == 200:
            probability = body["subscribe_probability"]
            a, b = st.columns(2)
            a.metric("Probability of subscribing", f"{probability:.1%}")
            b.metric("Recommendation", "Call" if body["call"] else "Do not call")
            st.progress(min(probability, 1.0))
            st.caption(
                f"Threshold {body['threshold']} from model {body['model_version']}. "
                "The service decided, not this page."
            )
        else:
            # A 422 is the service refusing a request it does not understand.
            # Showing the field it named is more use than showing the status.
            st.error(f"The service rejected this request with {code}.")
            for problem in body.get("detail", []):
                st.write(f"`{'.'.join(str(p) for p in problem['loc'])}`: {problem['msg']}")

with many:
    st.write(
        "Upload a CSV with one row per prospect and the same column names the "
        "service expects, or score the built in sample."
    )
    uploaded = st.file_uploader("CSV", type="csv")
    sample = pd.DataFrame([
        dict(payload, age=49, previous=3, days_since_last_contact=6, job="admin.",
             education="high.school", housing="no", loan="no", month="oct",
             day_of_week="wed", poutcome="success"),
        dict(payload, age=32, previous=0, days_since_last_contact=None,
             job="student", education="high.school", month="mar", poutcome="nonexistent"),
        dict(payload, age=58, previous=0, days_since_last_contact=None,
             job="blue-collar", education="basic.9y", housing="yes", loan="yes",
             month="jul", poutcome="nonexistent"),
    ])
    prospects = pd.read_csv(uploaded) if uploaded is not None else sample
    st.dataframe(prospects, width="stretch")

    if st.button("Score the list"):
        code, body = post(api, "/predict/batch", to_records(prospects))
        if code != 200:
            st.error(f"The service rejected the batch with {code}.")
            st.json(body)
        else:
            scored = prospects.copy()
            scored["probability"] = [row["subscribe_probability"] for row in body]
            scored["call"] = [row["call"] for row in body]
            scored = scored.sort_values("probability", ascending=False)
            st.dataframe(
                scored[["age", "job", "month", "poutcome", "probability", "call"]],
                width="stretch",
            )
            st.caption(
                f"{int(scored['call'].sum())} of {len(scored)} clear the "
                f"service's threshold of {threshold}."
            )

with about:
    st.write("Reported by the service, not copied into this page.")
    metrics = card["metrics"]
    a, b, c = st.columns(3)
    a.metric("Test ROC AUC", metrics["test_roc_auc"])
    b.metric(f"Test F1 at {threshold}", metrics["test_f1_at_threshold"])
    c.metric("Test F1 at 0.50", metrics["test_f1_at_half"])
    st.caption(
        "The two F1 figures are the same model at two cut points. The gap is "
        "what the stored threshold is worth."
    )
    st.write(f"**Study** {card['study']['trials']} trials, "
             f"best cross validated ROC AUC {card['study']['best_cv_roc_auc']}")
    st.write(f"**Trained** {card['trained_at']}")
    st.json(card["git"])
