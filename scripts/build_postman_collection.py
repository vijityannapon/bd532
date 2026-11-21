"""Generate postman/credit_risk_api.postman_collection.json (Section 4.2).

Run:  python scripts/build_postman_collection.py
Then import the JSON into Postman (or run with `newman run ...`).
"""
import json
from pathlib import Path

BASE = {"contract_type": "Cash loans", "gender": "F", "own_car": False, "own_realty": True, "children": 1,
        "income": 180000, "credit_amount": 500000, "annuity": 25000, "goods_price": 450000,
        "income_type": "Working", "education": "Higher education", "family_status": "Married",
        "housing_type": "House / apartment", "age_years": 38, "years_employed": 6.5,
        "occupation": "Core staff", "family_members": 3, "region_rating": 2,
        "ext_source_1": None, "ext_source_2": 0.62, "ext_source_3": 0.55}
LOW = dict(BASE, income=420000, annuity=22000, credit_amount=600000, goods_price=550000, age_years=46,
           years_employed=15, income_type="State servant", region_rating=1, family_members=2, children=0,
           ext_source_1=0.75, ext_source_2=0.78, ext_source_3=0.80)
HIGH = dict(BASE, gender="M", income=110000, credit_amount=900000, annuity=52000, goods_price=850000,
            education="Lower secondary", family_status="Single / not married", housing_type="Rented apartment",
            age_years=24, years_employed=0.5, occupation="Low-skill Laborers", region_rating=3, children=2,
            ext_source_2=0.12, ext_source_3=0.10, own_realty=False)


def req(name, method, path, body=None, tests=(), raw=None, prerequest=None, description=""):
    item = {"name": name, "request": {"method": method, "header": [{"key": "Content-Type", "value": "application/json"}],
                                      "url": {"raw": "{{baseUrl}}" + path, "host": ["{{baseUrl}}"],
                                              "path": [p for p in path.strip("/").split("/") if p]},
                                      "description": description}}
    if body is not None or raw is not None:
        item["request"]["body"] = {"mode": "raw", "raw": raw if raw is not None else json.dumps(body, indent=2)}
    events = []
    if prerequest:
        events.append({"listen": "prerequest", "script": {"type": "text/javascript", "exec": prerequest}})
    if tests:
        events.append({"listen": "test", "script": {"type": "text/javascript", "exec": list(tests)}})
    if events:
        item["event"] = events
    return item


def status(code):
    return f'pm.test("status is {code}", () => pm.response.to.have.status({code}));'


def err(code):
    return f'pm.test("error code {code}", () => pm.expect(pm.response.json().error.code).to.eql("{code}"));'


band = lambda b: f'pm.test("band is {b}", () => pm.expect(pm.response.json().prediction.decision_band).to.eql("{b}"));'
common_pred = [
    'const j = pm.response.json();',
    'pm.test("PD within [0,1]", () => pm.expect(j.prediction.probability_of_default).to.be.within(0, 1));',
    'pm.test("has reasons", () => pm.expect(j.explanation.reasons.length).to.be.above(0));',
    'pm.test("latency < 500 ms", () => pm.expect(pm.response.responseTime).to.be.below(500));',
    'pm.collectionVariables.set("lastRequestId", j.request_id);',
]

items = [
    {"name": "1. Operations", "item": [
        req("Health check", "GET", "/health", tests=[status(200),
            'pm.test("model loaded", () => pm.expect(pm.response.json().model_loaded).to.be.true);']),
        req("Model info", "GET", "/v1/model", tests=[status(200),
            'pm.test("gender not a model feature", () => pm.expect(pm.response.json().features).to.not.include("gender"));']),
    ]},
    {"name": "2. Prediction scenarios", "item": [
        req("S1 - Low-risk applicant", "POST", "/v1/predict", {"application_id": "PM-LOW", "applicant": LOW},
            [status(200), *common_pred, band("LOW_RISK")], description="Stable civil servant, strong bureau scores."),
        req("S2 - Typical applicant", "POST", "/v1/predict", {"application_id": "PM-MID", "applicant": BASE},
            [status(200), *common_pred], description="Average profile - band depends on model."),
        req("S3 - High-risk applicant", "POST", "/v1/predict", {"application_id": "PM-HIGH", "applicant": HIGH},
            [status(200), *common_pred, band("HIGH_RISK")], description="Young, short tenure, DSTI 47%, weak scores."),
        req("S4 - Fairness: same applicant, gender M", "POST", "/v1/predict",
            {"application_id": "PM-GENDER", "applicant": dict(BASE, gender="M")},
            [status(200), 'pm.test("same PD as S2 (gender unused)", () => pm.expect(pm.response.json().prediction.probability_of_default).to.eql(pm.collectionVariables.get("pdS2") * 1 || pm.response.json().prediction.probability_of_default));']),
        req("S5 - Batch of 3", "POST", "/v1/predict/batch",
            {"applications": [{"applicant": LOW}, {"applicant": BASE}, {"applicant": HIGH, "explain": False}]},
            [status(200), 'pm.test("3 results", () => pm.expect(pm.response.json().count).to.eql(3));']),
    ]},
    {"name": "3. Error handling", "item": [
        req("E1 - Missing required field -> 422", "POST", "/v1/predict",
            {"applicant": {k: v for k, v in BASE.items() if k != "income"}},
            [status(422), err("VALIDATION_ERROR"),
             'pm.test("points at income", () => pm.expect(pm.response.json().error.details[0].field).to.eql("applicant.income"));']),
        req("E2 - Invalid enum value -> 422", "POST", "/v1/predict", {"applicant": dict(BASE, education="PhD")},
            [status(422), err("VALIDATION_ERROR")]),
        req("E3 - Out-of-range value -> 422", "POST", "/v1/predict", {"applicant": dict(BASE, ext_source_2=1.7)},
            [status(422), err("VALIDATION_ERROR")]),
        req("E4 - Malformed JSON -> 400", "POST", "/v1/predict", raw='{"applicant": {"income": 1000,',
            tests=[status(400), err("MALFORMED_JSON")]),
        req("E5 - Inconsistent data (annuity > loan) -> 400", "POST", "/v1/predict",
            {"applicant": dict(BASE, annuity=900000)}, [status(400), err("BUSINESS_RULE_VIOLATION")]),
        req("E6 - Batch too large -> 413", "POST", "/v1/predict/batch", raw="{{bigBatch}}",
            prerequest=[f'const a = {json.dumps(BASE)};',
                        'pm.collectionVariables.set("bigBatch", JSON.stringify({applications: Array(101).fill({applicant: a, explain: false})}));'],
            tests=[status(413), err("BATCH_TOO_LARGE")]),
        req("E7 - Unknown endpoint -> 404", "GET", "/v1/unknown", tests=[status(404), err("NOT_FOUND")]),
        req("E8 - Wrong method -> 405", "GET", "/v1/predict", tests=[status(405), err("METHOD_NOT_ALLOWED")]),
    ]},
    {"name": "4. Feedback loop", "item": [
        req("F1 - Officer disagrees with model", "POST", "/v1/feedback",
            raw=json.dumps({"request_id": "{{lastRequestId}}", "application_id": "PM-HIGH",
                            "officer_decision": "APPROVE", "agrees_with_model": False,
                            "reason_code": "COLLATERAL_STRONG", "comment": "Guarantor provided"}, indent=2),
            tests=[status(201), 'pm.test("stored", () => pm.expect(pm.response.json().stored).to.be.true);']),
        req("F2 - Invalid feedback -> 422", "POST", "/v1/feedback",
            {"request_id": "x", "officer_decision": "MAYBE", "agrees_with_model": True},
            [status(422), err("VALIDATION_ERROR")]),
    ]},
]
# store S2's PD so S4 can compare against it
items[1]["item"][1]["event"][0]["script"]["exec"].append(
    'pm.collectionVariables.set("pdS2", j.prediction.probability_of_default);')
items[1]["item"][3]["event"][0]["script"]["exec"][1] = (
    'pm.test("same PD as S2 (gender is not used)", () => pm.expect(pm.response.json().prediction.probability_of_default)'
    '.to.eql(Number(pm.collectionVariables.get("pdS2"))));')

collection = {
    "info": {"name": "BD532 Credit Risk API", "description": "UI-Model interface tests (Section 4.2). "
             "Run the folders in order; S4 and F1 use variables set by earlier requests.",
             "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"},
    "variable": [{"key": "baseUrl", "value": "http://localhost:8000"}, {"key": "lastRequestId", "value": ""},
                 {"key": "pdS2", "value": ""}, {"key": "bigBatch", "value": ""}],
    "item": items,
}
out = Path(__file__).resolve().parents[1] / "postman" / "credit_risk_api.postman_collection.json"
out.write_text(json.dumps(collection, indent=2, ensure_ascii=False), encoding="utf-8")
print("Wrote", out)
