"""Credit Risk REST API (Section 3.6) + loan-officer web UI (Section 4.2).

Run:   uvicorn api.main:app --reload --port 8000
Docs:  http://localhost:8000/docs   (OpenAPI / Swagger UI)
UI:    http://localhost:8000/

Endpoints
---------
GET  /health            liveness + model status
GET  /v1/model          model metadata, thresholds, metrics
POST /v1/predict        score one application (+ SHAP reasons)
POST /v1/predict/batch  score up to MAX_BATCH applications
POST /v1/feedback       loan-officer feedback (human-in-the-loop)
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from credit_risk import config

from .schemas import (BatchPredictRequest, BatchPredictResponse, ErrorResponse, FeedbackRequest,
                      FeedbackResponse, HealthResponse, PredictRequest, PredictResponse)
from .service import BusinessRuleError, ModelService

MAX_BATCH = 100
STATIC_DIR = Path(__file__).parent / "static"
FEEDBACK_PATH = Path(os.getenv("FEEDBACK_PATH", config.FEEDBACK_PATH))
PREDICTION_LOG = Path(os.getenv("PREDICTION_LOG", config.DATA_DIR / "feedback" / "predictions.jsonl"))
log = logging.getLogger("credit_risk.api")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details=None):
        self.status, self.code, self.message, self.details = status, code, message, details


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.service = ModelService()
        log.info("Model loaded: %s", app.state.service.info)
    except Exception as exc:  # keep API up so /health can report the problem
        app.state.service = None
        log.error("Model could not be loaded: %s", exc)
    yield


app = FastAPI(
    title="Credit Risk Assessment API",
    version="1.0.0",
    description="BD532 Term Project - probability of default with SHAP explanations. "
                "The model gives a recommendation; the loan officer makes the decision.",
    lifespan=lifespan,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
               500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)


# --------------------------------------------------------------------------- #
# Middleware & error handling
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


def _error(request: Request, status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "details": details},
            "request_id": getattr(request.state, "request_id", None)}
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return _error(request, exc.status, exc.code, exc.message, exc.details)


@app.exception_handler(BusinessRuleError)
async def business_rule_handler(request: Request, exc: BusinessRuleError):
    return _error(request, 400, "BUSINESS_RULE_VIOLATION", str(exc), exc.details)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if any(e.get("type") == "json_invalid" for e in errors):
        return _error(request, 400, "MALFORMED_JSON", "Request body is not valid JSON")
    details = [{"field": ".".join(str(p) for p in e["loc"][1:]), "issue": e["msg"], "type": e["type"]}
               for e in errors]
    return _error(request, 422, "VALIDATION_ERROR", "Request payload failed validation", details)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    codes = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 415: "UNSUPPORTED_MEDIA_TYPE"}
    return _error(request, exc.status_code, codes.get(exc.status_code, "HTTP_ERROR"), str(exc.detail))


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    log.exception("Unhandled error")
    return _error(request, 500, "INTERNAL_ERROR", "Unexpected server error")


def _service(request: Request) -> ModelService:
    svc = request.app.state.service
    if svc is None:
        raise ApiError(503, "MODEL_UNAVAILABLE", "Model is not loaded. Train it with "
                       "`python -m credit_risk.train` or set MODEL_PATH.")
    return svc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _score(svc: ModelService, req: PredictRequest, request_id: str) -> dict:
    t0 = time.perf_counter()
    applicant = req.applicant.model_dump()
    out = svc.predict(applicant, explain=req.explain, top_k=req.top_k)
    resp = {"request_id": request_id, "application_id": req.application_id, "model": svc.info,
            **out, "latency_ms": round((time.perf_counter() - t0) * 1000, 2), "timestamp": _now()}
    # Monitoring log: no personal data, only what is needed to track drift/outcomes.
    _append_jsonl(PREDICTION_LOG, {"request_id": request_id, "application_id": req.application_id,
                                   "model_version": svc.info["version"], "timestamp": resp["timestamp"],
                                   "pd": out["prediction"]["probability_of_default"],
                                   "band": out["prediction"]["decision_band"],
                                   "gender": applicant.get("gender"), "warnings": len(out["warnings"])})
    return resp


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request):
    svc = request.app.state.service
    return {"status": "ok" if svc else "degraded", "model_loaded": svc is not None,
            "model_version": svc.info["version"] if svc else None}


@app.get("/v1/model", tags=["model"])
def model_info(request: Request):
    return _service(request).metadata()


@app.post("/v1/predict", response_model=PredictResponse, tags=["model"])
def predict(req: PredictRequest, request: Request):
    return _score(_service(request), req, request.state.request_id)


@app.post("/v1/predict/batch", response_model=BatchPredictResponse, tags=["model"])
def predict_batch(req: BatchPredictRequest, request: Request):
    if len(req.applications) > MAX_BATCH:
        raise ApiError(413, "BATCH_TOO_LARGE", f"Maximum {MAX_BATCH} applications per request",
                       {"received": len(req.applications)})
    svc = _service(request)
    results = [_score(svc, r, f"{request.state.request_id}-{i}") for i, r in enumerate(req.applications)]
    return {"count": len(results), "results": results}


@app.post("/v1/feedback", response_model=FeedbackResponse, status_code=201, tags=["feedback"])
def feedback(req: FeedbackRequest):
    record = {"feedback_id": uuid.uuid4().hex, "timestamp": _now(), **req.model_dump()}
    _append_jsonl(FEEDBACK_PATH, record)
    return {"feedback_id": record["feedback_id"], "stored": True, "timestamp": record["timestamp"]}


# --------------------------------------------------------------------------- #
# Web UI
# --------------------------------------------------------------------------- #
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC_DIR / "index.html")
