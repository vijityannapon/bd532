# CreditLens — Explainable AI-assisted Loan Default Risk Assessment

BD532 Term Project · ระบบ AI ช่วยเจ้าหน้าที่สินเชื่อประเมินความเสี่ยงผิดนัดชำระ พร้อมเหตุผลที่อธิบายได้ (SHAP)
ระบบมีการตรวจความเป็นธรรม ติดตามการทดลองด้วย MLflow และให้บริการผ่าน REST API กับ Web UI

> **AI ให้คำแนะนำ เจ้าหน้าที่เป็นผู้ตัดสินใจ** (Prompt + Annotate ไม่ใช่ Automate)

## Contents

| Section | เอกสาร | โค้ด |
|---|---|---|
| 1. Project Proposal | [docs/01_project_proposal.md](docs/01_project_proposal.md) | — |
| 2. System Design (goals, workflow, REQ/SPEC/ENV, ML Canvas, FTA/MCS, UX, deployment) | [docs/02_system_design.md](docs/02_system_design.md) | — |
| 3. ML Implementation (EDA, training, fairness, MLflow, SHAP, API) | [docs/03_ml_implementation.md](docs/03_ml_implementation.md) | `src/credit_risk/`, `api/` |
| 4. UI-Model Interface (payloads, errors, sequence diagram, tests) | [docs/04_ui_model_interface.md](docs/04_ui_model_interface.md) | `api/`, `postman/`, `scripts/` |
| 5. Conclusion, reflection, path to IS | [docs/05_conclusion.md](docs/05_conclusion.md) | — |
| **Report (PDF)** | [report/BD532_TermProject_Report.pdf](report/BD532_TermProject_Report.pdf) | `scripts/build_report.py` |
| **Slides (PDF)** | [report/BD532_TermProject_Slides.pdf](report/BD532_TermProject_Slides.pdf) (26 slides, speaker notes in `slides/slides/*.html` `<aside>`) | `slides/` |
| Generated results | [reports/results.md](reports/results.md), [reports/eda/eda_summary.md](reports/eda/eda_summary.md), [reports/interface_test_results.md](reports/interface_test_results.md) | |

## Project structure

```
bd532-credit-risk/
├── src/credit_risk/
│   ├── config.py        # paths, feature schema, cost/threshold policy
│   ├── data.py          # load Home Credit CSV, synthetic sample generator
│   ├── features.py      # preprocessing + domain feature engineering
│   ├── model.py         # sklearn pipelines + 6 experiment definitions + reweighing
│   ├── evaluate.py      # AUC/PR-AUC/KS/Brier, cost-based threshold, decision bands, risk score
│   ├── fairness.py      # group metrics: DI, demographic parity, equal opportunity, FPR gap
│   ├── explain.py       # SHAP global/local explanations in plain language (EN/TH)
│   ├── eda.py           # exploratory data analysis report
│   └── train.py         # training loop, validation, MLflow tracking, model selection, export
├── api/
│   ├── main.py          # FastAPI app: endpoints, error handling, middleware
│   ├── schemas.py       # request/response contracts (Pydantic → OpenAPI)
│   ├── service.py       # model loading, business rules, prediction + explanation
│   └── static/index.html# loan-officer web UI
├── tests/               # 40 pytest tests (features, model, SHAP, fairness, API)
├── postman/             # Postman collection (17 requests with test scripts)
├── scripts/             # Postman builder, live interface test runner, fairness chart, report builder
├── data/sample/         # synthetic sample (same schema as Home Credit)
├── data/raw/            # put application_train.csv here (not committed)
├── models/              # exported model bundle (.joblib)
├── reports/             # generated EDA, results, figures, interface test results
├── docs/                # report sections 1–5 (+ images/ screenshots)
└── report/              # final PDF report
```

## Setup

ต้องใช้ Python 3.10 ขึ้นไป (ทดสอบแล้วบน 3.12 และ 3.14)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Data

**Real data (ใช้สำหรับรายงาน):** Kaggle — [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)

1. Login Kaggle แล้วกด *I Understand and Accept* ในหน้า Rules ของ competition
2. ดาวน์โหลด `application_train.csv` (~160 MB) แล้ววางที่ `data/raw/application_train.csv`
   หรือใช้ Kaggle CLI: `kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data/raw` แล้ว unzip

ใช้เพียง 23 คอลัมน์จากไฟล์นี้ (ดู `config.RAW_TO_APPLICANT`)

**Sample data:** `data/sample/application_sample.csv` เป็นข้อมูลสังเคราะห์ 20,000 แถวที่ schema และหน่วยเหมือนของจริง
ถ้าไม่มีไฟล์จริง ระบบจะใช้ไฟล์นี้อัตโนมัติ (ผลลัพธ์ใช้สาธิตเท่านั้น) สร้างใหม่ได้ด้วย:

```bash
python -m credit_risk.data --rows 20000
```

## Run

```bash
# 1) Exploratory data analysis  -> reports/eda/
python -m credit_risk.eda

# 2) Train & compare 6 model versions, log to MLflow, export best model -> models/, reports/
python -m credit_risk.train
#    quick run on a subset of the real data:
python -m credit_risk.train --nrows 50000 --no-cv

# 3) MLflow Tracking UI  -> http://localhost:5001  (macOS uses port 5000 for AirPlay)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# 4) Start the API + web UI  -> http://localhost:8000  (Swagger: /docs)
uvicorn api.main:app --port 8000

# 5) Tests
pytest                                   # unit + API contract tests
python scripts/run_interface_tests.py    # live scenarios against the running server
```

Postman: import `postman/credit_risk_api.postman_collection.json` แล้วรัน collection (`baseUrl` = `http://localhost:8000`)
ถ้าแก้ scenario ให้ regenerate ด้วย `python scripts/build_postman_collection.py`

## Build the PDF report

ต้องมี Google Chrome และอินเทอร์เน็ต (สำหรับโหลด font Sarabun และ mermaid.js)

```bash
pip install -r requirements-report.txt
python scripts/plot_fairness.py          # after training: fairness comparison chart
python scripts/build_report.py --author "ชื่อ นามสกุล" --student-id 6XXXXXXXX \
    --github https://github.com/<user>/bd532-credit-risk
```
ผลลัพธ์อยู่ที่ `report/BD532_TermProject_Report.pdf` โดยเนื้อหามาจาก `docs/01–05` ถ้าแก้ docs แล้วให้รันคำสั่งนี้ใหม่

## API quick reference

| Method | Path | Description |
|---|---|---|
| GET | `/health` | liveness + model status |
| GET | `/v1/model` | model name/version, features, thresholds, metrics |
| POST | `/v1/predict` | PD, risk score, decision band, SHAP reasons, warnings |
| POST | `/v1/predict/batch` | up to 100 applications |
| POST | `/v1/feedback` | loan-officer feedback / outcome |

```bash
curl -s localhost:8000/v1/predict -H 'Content-Type: application/json' -d '{
  "applicant": {"contract_type":"Cash loans","own_car":false,"own_realty":true,"children":0,
    "income":180000,"credit_amount":500000,"annuity":25000,"goods_price":450000,
    "income_type":"Working","education":"Higher education","family_status":"Married",
    "housing_type":"House / apartment","age_years":38,"years_employed":6.5,
    "occupation":"Core staff","family_members":2,"region_rating":2,
    "ext_source_2":0.62,"ext_source_3":0.55}}'
```

รายละเอียด payload และ error codes ดูได้ที่ [docs/04_ui_model_interface.md](docs/04_ui_model_interface.md)

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/credit_risk_model.joblib` | model bundle to serve |
| `FEEDBACK_PATH` | `data/feedback/feedback.jsonl` | feedback store |
| `PREDICTION_LOG` | `data/feedback/predictions.jsonl` | monitoring log (no personal data except gender for fairness audit) |

นโยบายธุรกิจ (cost FN:FP = 5:1, review band = 50% ของ threshold) แก้ได้ใน `src/credit_risk/config.py`

## Key design decisions

* **Gender is never a model input**: ใช้เพื่อ audit fairness เท่านั้น ทดสอบด้วย Postman S4 และ `test_gender_does_not_change_prediction`
* **Preprocessing inside the sklearn Pipeline** เพื่อไม่ให้เกิด training–serving skew
* **Model selection** ใช้ calibration gate (Brier) → ROC-AUC → fairness (gender FPR gap) → latency
* **Cost-based threshold** (FN แพงกว่า FP 5 เท่า) แบ่งเป็น 3 band โดยไม่มีการปฏิเสธอัตโนมัติ
