"""UI-Model interface tests (Section 4.2) using FastAPI's TestClient."""
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(bundle_path, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("logs")
    os.environ["MODEL_PATH"] = str(bundle_path)
    import api.main as main
    main.FEEDBACK_PATH = tmp / "feedback.jsonl"
    main.PREDICTION_LOG = tmp / "predictions.jsonl"
    with TestClient(main.app) as c:
        c.feedback_path = main.FEEDBACK_PATH
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_model_info(client):
    r = client.get("/v1/model").json()
    assert r["thresholds"]["high_risk"] > r["thresholds"]["review"] > 0
    assert "gender" not in r["features"]


def test_predict_happy_path(client, applicant):
    r = client.post("/v1/predict", json={"application_id": "T-1", "applicant": applicant, "top_k": 4})
    assert r.status_code == 200
    body = r.json()
    assert r.headers["X-Request-ID"] == body["request_id"]
    p = body["prediction"]
    assert 0 <= p["probability_of_default"] <= 1
    assert p["decision_band"] in {"LOW_RISK", "REVIEW", "HIGH_RISK"}
    assert len(body["explanation"]["reasons"]) == 4


def test_gender_does_not_change_prediction(client, applicant):
    pf = client.post("/v1/predict", json={"applicant": dict(applicant, gender="F")}).json()
    pm = client.post("/v1/predict", json={"applicant": dict(applicant, gender="M")}).json()
    assert pf["prediction"]["probability_of_default"] == pm["prediction"]["probability_of_default"]


def test_risk_increases_with_worse_profile(client, applicant):
    good = client.post("/v1/predict", json={"applicant": dict(applicant, ext_source_2=0.9, ext_source_3=0.9)}).json()
    bad = client.post("/v1/predict", json={"applicant": dict(applicant, ext_source_2=0.05, ext_source_3=0.05)}).json()
    assert bad["prediction"]["probability_of_default"] > good["prediction"]["probability_of_default"]


def test_explain_false_omits_reasons(client, applicant):
    r = client.post("/v1/predict", json={"applicant": applicant, "explain": False}).json()
    assert r["explanation"] is None


def test_missing_required_field_422(client, applicant):
    a = dict(applicant); a.pop("income")
    r = client.post("/v1/predict", json={"applicant": a})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "VALIDATION_ERROR" and err["details"][0]["field"] == "applicant.income"


@pytest.mark.parametrize("field,value", [("education", "PhD"), ("age_years", 12), ("ext_source_2", 1.5),
                                         ("income", -5), ("unknown_field", 1)])
def test_invalid_values_422(client, applicant, field, value):
    r = client.post("/v1/predict", json={"applicant": dict(applicant, **{field: value})})
    assert r.status_code == 422


def test_malformed_json_400(client):
    r = client.post("/v1/predict", content=b"{bad json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "MALFORMED_JSON"


def test_business_rule_400(client, applicant):
    r = client.post("/v1/predict", json={"applicant": dict(applicant, annuity=900000)})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "BUSINESS_RULE_VIOLATION"


def test_out_of_range_warning(client, applicant):
    r = client.post("/v1/predict", json={"applicant": dict(applicant, income=9e7)}).json()
    assert any("income" in w for w in r["warnings"])


def test_batch(client, applicant):
    r = client.post("/v1/predict/batch", json={"applications": [{"applicant": applicant}] * 3})
    assert r.status_code == 200 and r.json()["count"] == 3


def test_batch_too_large_413(client, applicant):
    r = client.post("/v1/predict/batch", json={"applications": [{"applicant": applicant, "explain": False}] * 101})
    assert r.status_code == 413 and r.json()["error"]["code"] == "BATCH_TOO_LARGE"


def test_feedback_stored(client):
    r = client.post("/v1/feedback", json={"request_id": "abcdef123456", "officer_decision": "APPROVE",
                                          "agrees_with_model": False, "reason_code": "INCOME_VERIFIED"})
    assert r.status_code == 201 and r.json()["stored"] is True
    assert client.feedback_path.read_text().count("abcdef123456") == 1


def test_unknown_route_404(client):
    r = client.get("/v1/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Credit Risk Assistant" in r.text


def test_model_unavailable_503(client, applicant):
    svc = client.app.state.service
    client.app.state.service = None
    try:
        r = client.post("/v1/predict", json={"applicant": applicant})
        assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_UNAVAILABLE"
        assert client.get("/health").json()["status"] == "degraded"
    finally:
        client.app.state.service = svc
