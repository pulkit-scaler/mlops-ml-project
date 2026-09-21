"""The request and response contract.

Every allowed value below is a value the fitted OneHotEncoder actually saw
during training. That is not a coincidence and it is not maintained by hand
discipline: tests/test_contract.py reads artifacts/model_meta.json and fails if
this file and the encoder ever disagree.

Writing the categories out as Literal is the difference between a service that
rejects "Month to month" with a 422 and a service that accepts it, encodes it
as a row of zeros, and returns a confident number nobody can question.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["No", "Yes"]
YesNoPhone = Literal["No", "No phone service", "Yes"]
YesNoNet = Literal["No", "No internet service", "Yes"]


class Customer(BaseModel):
    """One customer, named exactly as the training columns are named.

    extra="forbid" turns an unrecognised field into a 422 instead of a silent
    drop. A caller who sends "Contract_Type" has a bug, and finding it in their
    integration tests is cheaper than finding it in a churn report six weeks on.
    """

    model_config = ConfigDict(extra="forbid")

    tenure: int = Field(ge=0, le=120, description="Whole months on the books")
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(
        ge=0,
        le=100_000,
        description="Billed to date. Zero for a customer who has not been billed yet.",
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    PhoneService: YesNo
    MultipleLines: YesNoPhone
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNet
    OnlineBackup: YesNoNet
    DeviceProtection: YesNoNet
    TechSupport: YesNoNet
    StreamingTV: YesNoNet
    StreamingMovies: YesNoNet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]


class Prediction(BaseModel):
    """What a caller may rely on. Anything not listed here is not promised.

    protected_namespaces is cleared because pydantic reserves the model_ prefix
    for its own attributes, and model_version is the name the rest of the
    industry uses.
    """

    model_config = ConfigDict(protected_namespaces=())

    churn_probability: float = Field(ge=0, le=1)
    churn: bool
    threshold: float = Field(ge=0, le=1)
    model_version: str


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    sklearn_version_matches: bool
