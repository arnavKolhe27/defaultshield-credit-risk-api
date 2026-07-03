# 🛡️ DefaultShield™: Live Credit Checkout Risk Engine

DefaultShield™ is a production-grade, high-throughput financial technology API Gateway. It combines an **Ensemble Machine Learning Committee (5-Fold LightGBM)** with a **Cloud NoSQL Logging Architecture (MongoDB Atlas)** to evaluate applicant default risk in real-time during e-commerce checkouts.

The system handles missing feature parameters natively, applies strict legal age constraints dynamically, and operates under a risk-optimized financial threshold ($0.7299$) to maximize automated checkout approvals while mitigating high-risk credit defaults.

---

## 🏗️ System Architecture & Data Pipeline

The gateway processes transactions across a multi-layered, non-blocking pipeline:

1. **Layer 1: Input Gatekeeper (Pydantic)** – Validates datatypes and strictly blocks applicants under 18 before data hits the ML models.
2. **Layer 2: Feature Synchronization (Pandas/NumPy)** – Aligns incoming payloads into the exact 41-feature shape required by LightGBM, safely substituting historical fallback medians for missing parameters.
3. **Layer 3: Consensus Inference (LightGBM Ensemble)** – Queries 5 distinct, cross-validated gradient-boosted trees to calculate an average consensus risk score.
4. **Layer 4: Cloud Logging Sync (MongoDB Atlas)** – Asynchronously offloads unstructured transaction payloads to a cloud data lake for audit trails without adding latency to the checkout experience.

---

## 🚀 Tech Stack

* **Backend Framework:** FastAPI (Python)
* **Asynchronous Server:** Uvicorn
* **Machine Learning Engine:** LightGBM, Joblib, Pandas, NumPy
* **Cloud Database:** MongoDB Atlas (via PyMongo & Certifi)
* **Environment Management:** Python Dotenv & OS Environment Pipelines

---

## 🛠️ Local Installation & Setup

### 1. Clone & Install Dependencies
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
pip install -r requirements.txt