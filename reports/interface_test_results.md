# UI-Model interface test results

Server: `http://localhost:8000`  |  17/17 passed  |  2026-11-21 20:55

| Group | Scenario | Method | Path | Expected | Actual | ms | Output | Result |
|---|---|---|---|---|---|---|---|---|
| 1. Operations | Health check | GET | /health | 200 | 200 | 3 | model_loaded=True | PASS |
| 1. Operations | Model info | GET | /v1/model | 200 | 200 | 1 | 28 features, gender used: False | PASS |
| 2. Prediction scenarios | S1 - Low-risk applicant | POST | /v1/predict | 200 | 200 | 101 | PD=1.3%, band=LOW_RISK | PASS |
| 2. Prediction scenarios | S2 - Typical applicant | POST | /v1/predict | 200 | 200 | 43 | PD=4.0%, band=LOW_RISK | PASS |
| 2. Prediction scenarios | S3 - High-risk applicant | POST | /v1/predict | 200 | 200 | 43 | PD=36.9%, band=HIGH_RISK | PASS |
| 2. Prediction scenarios | S4 - Fairness: same applicant, gender M | POST | /v1/predict | 200 | 200 | 42 | PD=4.0%, band=LOW_RISK, same PD as S2 | PASS |
| 2. Prediction scenarios | S5 - Batch of 3 | POST | /v1/predict/batch | 200 | 200 | 120 | 3 results | PASS |
| 3. Error handling | E1 - Missing required field -> 422 | POST | /v1/predict | 422 | 422 | 1 | VALIDATION_ERROR | PASS |
| 3. Error handling | E2 - Invalid enum value -> 422 | POST | /v1/predict | 422 | 422 | 1 | VALIDATION_ERROR | PASS |
| 3. Error handling | E3 - Out-of-range value -> 422 | POST | /v1/predict | 422 | 422 | 1 | VALIDATION_ERROR | PASS |
| 3. Error handling | E4 - Malformed JSON -> 400 | POST | /v1/predict | 400 | 400 | 1 | MALFORMED_JSON | PASS |
| 3. Error handling | E5 - Inconsistent data (annuity > loan) -> 400 | POST | /v1/predict | 400 | 400 | 1 | BUSINESS_RULE_VIOLATION | PASS |
| 3. Error handling | E6 - Batch too large -> 413 | POST | /v1/predict/batch | 413 | 413 | 2 | BATCH_TOO_LARGE | PASS |
| 3. Error handling | E7 - Unknown endpoint -> 404 | GET | /v1/unknown | 404 | 404 | 1 | NOT_FOUND | PASS |
| 3. Error handling | E8 - Wrong method -> 405 | GET | /v1/predict | 405 | 405 | 1 | METHOD_NOT_ALLOWED | PASS |
| 4. Feedback loop | F1 - Officer disagrees with model | POST | /v1/feedback | 201 | 201 | 1 | stored | PASS |
| 4. Feedback loop | F2 - Invalid feedback -> 422 | POST | /v1/feedback | 422 | 422 | 1 | VALIDATION_ERROR | PASS |
