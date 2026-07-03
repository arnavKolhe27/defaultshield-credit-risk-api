import os
import json
import logging
import joblib
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import certifi
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from pymongo import MongoClient

# Set up clean logging to audit requests to BOTH terminal and internal logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DefaultShieldGate")

app = FastAPI(
    title="DefaultShield™ Live Checkout Risk Gateway",
    description="Unified API Gateway combining input checking, MongoDB cloud logging, and ensemble model inference.",
    version="1.1.0"
)

# ---------------------------------------------------------------------
# LAYER 0: CLOUD MONGODB ATLAS CONNECTION CONFIGURATION
# ---------------------------------------------------------------------
# ⚠️ REPLACE WITH YOUR ACTUAL DATABASE USERNAME AND PASSWORD CONFIGURATION
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://PLACEHOLDER_IF_NOT_SET")
try:
    # Initialize connection using certifi to avoid local SSL certificate validation errors
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client["credit_risk_platform"]
    logs_collection = db["transaction_logs"]
    logger.info("📡 Successfully established persistent connection to Cloud MongoDB Atlas cluster!")
except Exception as mongo_err:
    logger.error(f"⚠️ MongoDB Connection Failure (System will run in fallback offline mode): {mongo_err}")
    logs_collection = None

# ---------------------------------------------------------------------
# LAYER 1: STARTUP ARTIFACT & CATEGORY DICTIONARY PARSING
# ---------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
models = []
blueprint = {}
category_vocabulary = {}

try:
    # 1. Load the core preprocessing metadata structural file
    blueprint = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocessing_blueprint.pkl"))
    
    # 2. Reload the 5 distinct LightGBM models into memory
    models = [lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, f"lgb_fold_{i+1}.txt")) for i in range(5)]
    
    # 3. FIXED: Extract categorical maps directly from the text model file to prevent pandas validation crashes
    model_1_path = os.path.join(ARTIFACTS_DIR, "lgb_fold_1.txt")
    extracted_vocab_lists = []
    
    if os.path.exists(model_1_path):
        with open(model_1_path, "r", encoding="utf-8") as f:
            for line in f:
                if "pandas_categorical:" in line:
                    try:
                        raw_json = line.split("pandas_categorical:")[1].strip()
                        extracted_vocab_lists = json.loads(raw_json)
                        break
                    except Exception as json_err:
                        logger.warning(f"Could not parse model metadata line: {json_err}")
                        
        # Pair feature names to their original categorical training ranges
        for index, cat_col_name in enumerate(blueprint.get("cat_cols", [])):
            if index < len(extracted_vocab_lists):
                category_vocabulary[cat_col_name] = extracted_vocab_lists[index]
                
    logger.info("🚀 Production engine state, model committee, and categorical dictionaries successfully loaded.")
except Exception as e:
    logger.critical(f"❌ Failed to load model state on startup. API cannot process risk. Error: {e}")

# ---------------------------------------------------------------------
# LAYER 2: THE INPUT GATEKEEPER (Strict Hard-Coded Validation)
# ---------------------------------------------------------------------
class CheckoutInputValidator(BaseModel):
    merchant_group: str = Field(..., description="The store group identifying where the user is buying items.")
    age: float = Field(..., description="Age of applicant. Must be a verified adult.")
    
    # Optional parameters (Defaults to None to safely catch cold-start / brand-new users)
    num_arch_ok_12_24m: float = Field(None, gte=0)
    account_worst_status_0_3m: float = Field(None, gte=0)
    avg_payment_span_0_12m: float = Field(None, gte=0)
    max_paid_inv_0_24m: float = Field(None, gte=0)
    num_unpaid_bills: float = Field(None, gte=0)

    # HARD RULE: Explicitly block anyone under 18 before it reaches the AI
    @field_validator('age')
    @classmethod
    def verify_legal_age(cls, value: float) -> float:
        if value < 18.0:
            raise ValueError("Applicant must be 18 years of age or older to qualify for credit options.")
        if value > 110.0:
            raise ValueError("Invalid age parameter provided.")
        return value

# ---------------------------------------------------------------------
# LAYER 3: INFERENCE ENGINE WITH DYNAMIC FEATURE-NAME SYNCHRONIZATION
# ---------------------------------------------------------------------
@app.post("/api/v1/evaluate-checkout", status_code=status.HTTP_200_OK)
async def evaluate_checkout_risk(payload: CheckoutInputValidator):
    if not models or not blueprint:
        raise HTTPException(status_code=503, detail="Risk models are currently uninitialized or missing.")
        
    raw_features = payload.model_dump()
    logger.info(f"Incoming Request: Age {raw_features['age']} | Store '{raw_features['merchant_group']}'")

    # EXTRACT THE EXACT FEATURE NAMES EXPECTED BY THE LIGHTGBM MODEL (ALL 41 COLUMNS)
    expected_features = models[0].feature_name()

    # Reconstruct a master record explicitly mapped to what the model demands
    master_record = {}
    for col in expected_features:
        # 1. If it's a value supplied in the API payload, use it
        if col in raw_features and raw_features[col] is not None:
            master_record[col] = raw_features[col]
        # 2. If it's a known numerical column but missing, use the historical fallback median
        elif col in blueprint["num_cols"]:
            master_record[col] = blueprint["fallback_medians"].get(col, 0.0)
        # 3. If it's a categorical column but missing, flag it as missing
        elif col in blueprint["cat_cols"]:
            master_record[col] = "Missing_Category"
        # 4. Fail-safe placeholder for any extra hidden columns to ensure shape matches perfectly
        else:
            master_record[col] = 0.0

    # Force the dataframe creation to use the exact 41-feature order
    input_df = pd.DataFrame([master_record], columns=expected_features)

    # Apply type casting strictly according to original column identities
    for col in expected_features:
        if col in blueprint["num_cols"]:
            input_df[col] = input_df[col].astype(np.float64)
        elif col in blueprint["cat_cols"]:
            train_vocabulary = category_vocabulary.get(col, None)
            if train_vocabulary:
                cat_type = pd.CategoricalDtype(categories=train_vocabulary, ordered=False)
                input_df[col] = input_df[col].astype(str).astype(cat_type)
            else:
                input_df[col] = input_df[col].astype(str).astype('category')
        else:
            # Fallback type coercion for unexpected columns
            input_df[col] = input_df[col].astype(np.float64)

    try:
        # Query our 5-model committee to calculate average consensus probability
        total_score = 0.0
        for model in models:
            total_score += model.predict(input_df)[0]
        final_risk_score = total_score / len(models)

        # Determine risk action based on our optimized threshold (0.7299)
        decision_threshold = blueprint["optimal_threshold"]
        final_decision = "DENY" if final_risk_score >= decision_threshold else "APPROVE"

        logger.info(f"Evaluation Complete. Columns processed: {len(expected_features)} | Score: {final_risk_score:.4f} -> Result: {final_decision}")

        response_body = {
            "status": "processed",
            "default_probability": round(float(final_risk_score), 5),
            "decision": final_decision,
            "policy_applied": f"F2 Risk-Optimized Guardrail (Threshold: {decision_threshold:.4f})"
        }

        # ---------------------------------------------------------------------
        # LAYER 4: ASYNCHRONOUS-STYLE MONGODB CLOUD LOGGING
        # ---------------------------------------------------------------------
        if logs_collection is not None:
            try:
                # Build unstructured NoSQL document mapping current input/output context
                transaction_document = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    # exclude_none=True optimizes document size and leverages NoSQL dynamic schemas
                    "inputs_received": payload.model_dump(exclude_none=True),
                    "outputs_generated": response_body
                }
                # Insert document straight to Atlas Cloud Cluster
                logs_collection.insert_one(transaction_document)
                logger.info("💾 Transaction log successfully synchronized with MongoDB Cloud.")
            except Exception as db_log_err:
                # Wrapped inside a safe try-catch block to guarantee that database lag 
                # never freezes or blocks the customer checkout experience.
                logger.error(f"⚠️ Non-blocking database logging failure: {db_log_err}")

        return response_body
        
    except Exception as e:
        logger.error(f"❌ Structural Model Processing Failure: {e}")
        raise HTTPException(status_code=500, detail=f"Inference processing error: {str(e)}")