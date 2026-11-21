"""Request / response contracts of the Credit Risk API (Section 4.1).

Pydantic models are the single source of truth for the payload structure:
FastAPI validates every request against them and publishes them as the
OpenAPI schema at /docs and /openapi.json.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ContractType = Literal["Cash loans", "Revolving loans"]
IncomeType = Literal["Working", "Commercial associate", "Pensioner", "State servant",
                     "Unemployed", "Student", "Businessman", "Maternity leave"]
Education = Literal["Secondary / secondary special", "Higher education", "Incomplete higher",
                    "Lower secondary", "Academic degree"]
FamilyStatus = Literal["Married", "Single / not married", "Civil marriage", "Separated", "Widow", "Unknown"]
HousingType = Literal["House / apartment", "With parents", "Municipal apartment",
                      "Rented apartment", "Office apartment", "Co-op apartment"]
Occupation = Literal["Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
                     "High skill tech staff", "Accountants", "Medicine staff", "Security staff",
                     "Cooking staff", "Cleaning staff", "Private service staff", "Low-skill Laborers",
                     "Waiters/barmen staff", "Secretaries", "Realty agents", "HR staff", "IT staff"]
DecisionBand = Literal["LOW_RISK", "REVIEW", "HIGH_RISK"]


class Applicant(BaseModel):
    """Loan application as entered by the loan officer (human units)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "contract_type": "Cash loans", "gender": "F", "own_car": False, "own_realty": True,
        "children": 1, "income": 180000, "credit_amount": 500000, "annuity": 25000,
        "goods_price": 450000, "income_type": "Working", "education": "Higher education",
        "family_status": "Married", "housing_type": "House / apartment", "age_years": 38,
        "years_employed": 6.5, "occupation": "Core staff", "family_members": 3,
        "region_rating": 2, "ext_source_1": None, "ext_source_2": 0.62, "ext_source_3": 0.55}})

    contract_type: ContractType
    gender: Optional[Literal["M", "F"]] = Field(
        None, description="Collected for fairness monitoring ONLY - never used by the model.")
    own_car: bool
    own_realty: bool
    children: int = Field(..., ge=0, le=20)
    income: float = Field(..., gt=0, le=1e8, description="Annual income")
    credit_amount: float = Field(..., gt=0, le=1e8, description="Requested loan amount")
    annuity: float = Field(..., gt=0, le=1e7, description="Loan installment (annuity)")
    goods_price: Optional[float] = Field(None, gt=0, le=1e8, description="Price of goods / collateral")
    income_type: IncomeType
    education: Education
    family_status: FamilyStatus
    housing_type: HousingType
    age_years: float = Field(..., ge=18, le=100)
    years_employed: Optional[float] = Field(None, ge=0, le=60, description="null = not employed / pensioner")
    occupation: Optional[Occupation] = None
    family_members: int = Field(..., ge=1, le=20)
    region_rating: int = Field(..., ge=1, le=3, description="1 = best region, 3 = worst")
    ext_source_1: Optional[float] = Field(None, ge=0, le=1, description="External bureau score (0-1)")
    ext_source_2: Optional[float] = Field(None, ge=0, le=1)
    ext_source_3: Optional[float] = Field(None, ge=0, le=1)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    application_id: Optional[str] = Field(None, max_length=64, examples=["APP-2026-000123"])
    applicant: Applicant
    explain: bool = Field(True, description="Return SHAP-based reasons")
    top_k: int = Field(5, ge=1, le=10, description="Number of reasons to return")


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applications: list[PredictRequest] = Field(..., min_length=1)


class Reason(BaseModel):
    feature: str
    label: str
    label_th: str
    value: str
    shap_value: float = Field(..., description="Contribution to log-odds of default")
    direction: Literal["increases_risk", "decreases_risk"]
    text: str


class Explanation(BaseModel):
    base_probability: float = Field(..., description="Average PD of the training population")
    reasons: list[Reason]


class Thresholds(BaseModel):
    review: float
    high_risk: float


class Prediction(BaseModel):
    probability_of_default: float = Field(..., ge=0, le=1)
    risk_score: int = Field(..., ge=0, le=1000, description="Higher = safer (PDO scaling)")
    decision_band: DecisionBand
    recommendation: str
    thresholds: Thresholds


class ModelRef(BaseModel):
    name: str
    version: str
    trained_at: str
    data_source: str


class PredictResponse(BaseModel):
    request_id: str
    application_id: Optional[str]
    model: ModelRef
    prediction: Prediction
    explanation: Optional[Explanation] = None
    warnings: list[str] = Field(default_factory=list,
                                description="Inputs outside the training range (1st-99th percentile)")
    latency_ms: float
    timestamp: str


class BatchPredictResponse(BaseModel):
    count: int
    results: list[PredictResponse]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(..., min_length=8, max_length=64)
    application_id: Optional[str] = Field(None, max_length=64)
    officer_decision: Literal["APPROVE", "DECLINE", "REFER"]
    agrees_with_model: bool
    reason_code: Optional[Literal["INCOME_VERIFIED", "COLLATERAL_STRONG", "DOCS_INCONSISTENT",
                                  "KNOWN_CUSTOMER", "POLICY_EXCEPTION", "MODEL_MISSED_RISK", "OTHER"]] = None
    comment: Optional[str] = Field(None, max_length=500)
    actual_outcome: Optional[Literal["REPAID", "DEFAULTED"]] = Field(
        None, description="Filled later by collections - becomes a new training label")


class FeedbackResponse(BaseModel):
    feedback_id: str
    stored: bool
    timestamp: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Optional[list | dict] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: Optional[str] = None
