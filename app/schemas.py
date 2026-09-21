"""The request and response contract.

Every allowed value below is a value the fitted OneHotEncoder actually saw
during training. That is not maintained by hand discipline:
tests/test_contract.py reads artifacts/model_meta.json and fails if this file
and the encoder ever disagree.

Note what the request does not contain. The five economic columns are not
properties of a customer, so the service supplies them. pdays is not here
either, because 999 meaning "never contacted" is an artifact of a CSV and has
no place in a public contract. Callers send days_since_last_contact, which is
null when there was none.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNoUnknown = Literal["no", "unknown", "yes"]


class Customer(BaseModel):
    """One prospect, as the calling system knows them.

    extra="forbid" turns an unrecognised field into a 422 instead of a silent
    drop. A caller who sends "day_of_the_week" has a bug, and finding it in
    their integration tests is cheaper than finding it in a campaign report.
    """

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=17, le=110)
    campaign: int = Field(ge=1, le=100, description="Contacts during this campaign")
    previous: int = Field(ge=0, le=100, description="Contacts before this campaign")
    days_since_last_contact: int | None = Field(
        default=None,
        ge=0,
        le=990,
        description="Null when the customer has never been contacted before",
    )

    job: Literal[
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ]
    marital: Literal["divorced", "married", "single", "unknown"]
    education: Literal[
        "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
        "professional.course", "university.degree", "unknown",
    ]
    default: YesNoUnknown
    housing: YesNoUnknown
    loan: YesNoUnknown
    contact: Literal["cellular", "telephone"]
    month: Literal["apr", "aug", "dec", "jul", "jun", "mar", "may", "nov", "oct", "sep"]
    day_of_week: Literal["fri", "mon", "thu", "tue", "wed"]
    poutcome: Literal["failure", "nonexistent", "success"]


class Prediction(BaseModel):
    """What a caller may rely on. Anything not listed here is not promised."""

    model_config = ConfigDict(protected_namespaces=())

    subscribe_probability: float = Field(ge=0, le=1)
    call: bool = Field(description="True when the probability clears the stored threshold")
    threshold: float = Field(ge=0, le=1)
    model_version: str


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    model_commit: str | None
    sklearn_version_matches: bool
