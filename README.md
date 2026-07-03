<div align="center">

# 🛡️ DefaultShield™ — Live Credit Checkout Risk Engine

### Production-Grade FastAPI Gateway for Real-Time Credit Default Risk Evaluation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/ML%20Engine-LightGBM%20Ensemble-brightgreen?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![MongoDB Atlas](https://img.shields.io/badge/Database-MongoDB%20Atlas-47A248?style=for-the-badge&logo=mongodb)](https://www.mongodb.com/atlas)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)]()

</div>

---

## 📖 Overview 

**DefaultShield™** is a production-grade, high-throughput financial technology API gateway built to evaluate applicant **credit default risk in real time** during e-commerce checkout flows.

It combines a **5-fold LightGBM ensemble committee** for consensus-based risk scoring with a **secure, non-blocking MongoDB Atlas logging layer** for full transaction audit trails — all served through a lightweight, asynchronous **FastAPI** gateway.

The system enforces strict legal-age constraints at the input layer, gracefully handles missing applicant data using historical fallback medians, and applies a **risk-optimized decision threshold (0.7299)** tuned to maximize automated checkout approvals while minimizing exposure to high-risk defaults.

---

## 🏗️ System Architecture & Data Pipeline

Every transaction flows through a multi-layered, non-blocking pipeline:

| Layer | Component | Responsibility |
|-------|-----------|-----------------|
| **Layer 1** | Input Gatekeeper (`Pydantic`) | Validates payload types and hard-blocks applicants under 18 before any data reaches the ML models |
| **Layer 2** | Feature Synchronization (`Pandas` / `NumPy`) | Aligns incoming payloads to the exact 41-feature shape required by LightGBM, substituting historical fallback medians for missing numeric fields and flagging missing categorical fields |
| **Layer 3** | Consensus Inference (`LightGBM` Ensemble) | Queries 5 independently cross-validated gradient-boosted tree models and averages their outputs into a single consensus risk score |
| **Layer 4** | Cloud Logging Sync (`MongoDB Atlas`) | Asynchronously persists unstructured transaction payloads (inputs + outputs) to a cloud data lake for audit trails, without adding latency to the checkout response |

```
Client Request
      │
      ▼
┌─────────────────────┐
│  Pydantic Validator  │  ← rejects malformed / underage applicants
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ Feature Synchronizer │  ← builds 41-feature record, imputes medians
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  5-Fold LightGBM     │  ← averages 5 model predictions
│  Ensemble Committee  │
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  Threshold Decision  │  ← APPROVE / DENY @ 0.7299
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  MongoDB Atlas Log   │  ← async, non-blocking audit trail
└─────────┬────────────┘
          ▼
     JSON Response
```

---

## ✨ Key Features

- ⚡ **Real-time inference** — sub-second risk scoring at checkout via FastAPI + Uvicorn
- 🧠 **Ensemble ML scoring** — 5 independently trained LightGBM folds averaged for a robust consensus probability, rather than relying on a single model
- 🔞 **Hard-coded legal safeguards** — applicants under 18 are rejected at the validation layer before any inference occurs
- 🧩 **Graceful degradation on missing data** — numeric fields fall back to historical training medians; categorical fields are flagged as `Missing_Category` rather than causing failures
- 🎯 **Risk-optimized decision threshold** — tuned to `0.7299` to balance approval throughput against default risk exposure
- ☁️ **Cloud-native audit logging** — every transaction (inputs + outputs) is persisted to MongoDB Atlas asynchronously, with logging failures isolated so they never block or slow down the checkout response
- 🛟 **Fail-safe startup & runtime handling** — the API degrades gracefully (falls back to offline mode) if the database connection fails, and returns clear HTTP errors if models fail to load

---

## 🚀 Tech Stack

| Category | Technology |
|-----------|------------|
| **Backend Framework** | [FastAPI](https://fastapi.tiangolo.com/) (Python) |
| **ASGI Server** | [Uvicorn](https://www.uvicorn.org/) |
| **Machine Learning Engine** | [LightGBM](https://lightgbm.readthedocs.io/) (5-fold ensemble), Joblib |
| **Data Processing** | Pandas, NumPy |
| **Cloud Database** | MongoDB Atlas (via PyMongo + Certifi for TLS/SSL) |
| **Validation** | Pydantic v2 (`BaseModel`, `field_validator`) |
| **Environment Management** | Python `os.environ` / `.env` |

---

## 📁 Project Structure

```
defaultshield-credit-risk-api/
├── artifacts/                     # Serialized model artifacts
│   ├── lgb_fold_1.txt              # LightGBM booster — fold 1
│   ├── lgb_fold_2.txt              # LightGBM booster — fold 2
│   ├── lgb_fold_3.txt              # LightGBM booster — fold 3
│   ├── lgb_fold_4.txt              # LightGBM booster — fold 4
│   ├── lgb_fold_5.txt              # LightGBM booster — fold 5
│   └── preprocessing_blueprint.pkl # Fallback medians, column types, optimal threshold
├── main.py                        # FastAPI application, inference pipeline, MongoDB logging
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

---

## 🛠️ Local Installation & Setup

### Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- A [MongoDB Atlas](https://www.mongodb.com/atlas) cluster (optional — the API runs in offline fallback mode if unavailable)
- `pip` for dependency management

### 1. Clone the repository

```bash
git clone https://github.com/arnavKolhe27/defaultshield-credit-risk-api.git
cd defaultshield-credit-risk-api
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file (or export directly in your shell) with your MongoDB Atlas connection string:

```env
MONGO_URI=mongodb+srv://<username>:<password>@<cluster-url>/?retryWrites=true&w=majority
```

> ⚠️ Never commit real credentials. Ensure `.env` is listed in `.gitignore`. If `MONGO_URI` is not set, the API automatically falls back to offline mode and continues serving predictions without logging.

### 5. Run the API

```bash
uvicorn main:app --reload
```

The API will be available at **http://127.0.0.1:8000**, with interactive Swagger docs at **http://127.0.0.1:8000/docs**.

---

## 📡 API Reference

### `POST /api/v1/evaluate-checkout`

Evaluates real-time credit default risk for a checkout applicant.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `merchant_group` | `string` | ✅ | The store/merchant group where the applicant is checking out |
| `age` | `float` | ✅ | Applicant age — must be ≥ 18 and ≤ 110 |
| `num_arch_ok_12_24m` | `float` | ❌ | Number of archived accounts in good standing (12–24 months) |
| `account_worst_status_0_3m` | `float` | ❌ | Worst account status in the last 0–3 months |
| `avg_payment_span_0_12m` | `float` | ❌ | Average payment span over the last 0–12 months |
| `max_paid_inv_0_24m` | `float` | ❌ | Maximum paid invoice amount over the last 0–24 months |
| `num_unpaid_bills` | `float` | ❌ | Current count of unpaid bills |

> Any omitted numeric field is automatically imputed using historical fallback medians from the training set; the model accepts 41 total engineered features under the hood.

**Example Request**

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/evaluate-checkout" \
  -H "Content-Type: application/json" \
  -d '{
        "merchant_group": "Electronics",
        "age": 29,
        "num_arch_ok_12_24m": 3,
        "account_worst_status_0_3m": 1,
        "avg_payment_span_0_12m": 14.5,
        "max_paid_inv_0_24m": 250.0,
        "num_unpaid_bills": 0
      }'
```

**Example Response**

```json
{
  "status": "processed",
  "default_probability": 0.18452,
  "decision": "APPROVE",
  "policy_applied": "F2 Risk-Optimized Guardrail (Threshold: 0.7299)"
}
```

**Error Responses**

| Status Code | Cause |
|--------------|-------|
| `422` | Invalid payload (e.g. applicant under 18, malformed field types) |
| `500` | Internal inference failure |
| `503` | Models failed to load at startup |

---

## 🔐 Security & Compliance Notes

- Applicants under 18 are rejected at the validation layer, before any model inference occurs.
- Database logging is fully isolated in a try/except block — a MongoDB outage or latency spike can **never** block or delay a checkout decision.
- TLS/SSL certificate validation for the MongoDB Atlas connection is handled via `certifi` to avoid local CA trust issues.
- No credentials are hard-coded in source; connection strings are read from environment variables at runtime.

---

## 🗺️ Roadmap

- [ ] Add authentication/API key middleware for merchant-level access control
- [ ] Add rate limiting for high-throughput checkout traffic
- [ ] Add automated test suite (unit + integration tests for the inference pipeline)
- [ ] Add CI/CD pipeline (GitHub Actions) for automated linting, testing, and deployment
- [ ] Add Dockerfile for containerized deployment

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes
   ```bash
   git commit -m "Add: your feature description"
   ```
4. Push to your branch and open a Pull Request

---

## 👤 Author

**Arnav Kolhe**
GitHub: [@arnavKolhe27](https://github.com/arnavKolhe27)

---

<div align="center">

**⭐ If you found this project useful, consider giving it a star on GitHub! ⭐**

</div>
