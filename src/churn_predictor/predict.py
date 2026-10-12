"""Model inference module for Customer Churn Prediction.

Handles customer feature extraction, categorical encoding, scaling,
and inference using the trained best XGBoost model artifact.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

# Precomputed scaling constants matching the training dataset
SCALER_PARAMS = {
    "tenure_max": 72.0,
    "monthly_charges_mean": 64.761692,
    "monthly_charges_std": 30.087911,
    "total_charges_mean": 2281.916928,
    "total_charges_std": 2265.109576,
    "cltv_mean": 4400.295755,
    "cltv_std": 1182.973161,
}

MODEL_FEATURE_NAMES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Paperless Billing",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
    "Multiple Lines_No phone service",
    "Multiple Lines_Yes",
    "Internet Service_Fiber optic",
    "Internet Service_No",
    "Online Security_No internet service",
    "Online Security_Yes",
    "Online Backup_No internet service",
    "Online Backup_Yes",
    "Device Protection_No internet service",
    "Device Protection_Yes",
    "Tech Support_No internet service",
    "Tech Support_Yes",
    "Streaming TV_No internet service",
    "Streaming TV_Yes",
    "Streaming Movies_No internet service",
    "Streaming Movies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "Payment Method_Credit card (automatic)",
    "Payment Method_Electronic check",
    "Payment Method_Mailed check",
]

_CACHED_MODEL = None


def load_model(model_path: Optional[Path | str] = None):
    """Load the trained model artifact from local disk or S3 fallback."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    candidates = [
        Path(__file__).parent / "models" / "best_churn_model.joblib",
        Path("src/churn_predictor/models/best_churn_model.joblib"),
        config.PROJECT_ROOT / "src/churn_predictor/models/best_churn_model.joblib",
        config.MODELS_DIR / "best_churn_model.joblib",
        Path("models/best_churn_model.joblib"),
    ]

    if model_path:
        candidates.insert(0, Path(model_path))

    for path in candidates:
        if path.exists():
            logger.info(f"Loading model artifact from {path}")
            _CACHED_MODEL = joblib.load(path)
            return _CACHED_MODEL

    # If model is not found locally, try downloading from S3
    try:
        import boto3

        s3_bucket = os.getenv("S3_BUCKET_NAME", config.S3_BUCKET)
        s3_key = "output/models/best_churn_model.joblib"
        target_path = Path("src/churn_predictor/models/best_churn_model.joblib")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Attempting to download model from s3://{s3_bucket}/{s3_key}")
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", config.AWS_REGION))
        s3.download_file(s3_bucket, s3_key, str(target_path))
        _CACHED_MODEL = joblib.load(target_path)
        logger.info(f"Successfully downloaded and loaded model from S3")
        return _CACHED_MODEL
    except Exception as e:
        logger.warning(f"Could not load model from S3 fallback: {e}")

    raise FileNotFoundError(
        f"Model artifact best_churn_model.joblib not found. Checked: {[str(p) for p in candidates]}"
    )


def _normalize_key(key: str) -> str:
    """Normalize input keys to allow snake_case or Title Case."""
    return key.strip().lower().replace("_", " ")


def preprocess_single_record(customer: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess a single raw customer dictionary into the exact 31 model features."""
    norm = {_normalize_key(k): v for k, v in customer.items()}

    def get_val(key_norm: str, default: Any = None) -> Any:
        return norm.get(key_norm, default)

    def to_binary(key_norm: str, default: int = 0) -> int:
        val = str(get_val(key_norm, "")).strip().lower()
        if val in ("yes", "1", "true", "y"):
            return 1
        if val in ("no", "0", "false", "n"):
            return 0
        return default

    # Gender: Female=0, Male=1
    gender_raw = str(get_val("gender", "Female")).strip().lower()
    gender = 1 if gender_raw == "male" else 0

    senior_citizen = to_binary("senior citizen")
    partner = to_binary("partner")
    dependents = to_binary("dependents")
    phone_service = to_binary("phone service", default=1)
    paperless_billing = to_binary("paperless billing", default=1)

    # Numerical features
    try:
        tenure_months = float(get_val("tenure months", 1.0))
    except (ValueError, TypeError):
        tenure_months = 1.0

    try:
        monthly_charges = float(get_val("monthly charges", 65.0))
    except (ValueError, TypeError):
        monthly_charges = 65.0

    try:
        total_charges = float(get_val("total charges", monthly_charges * max(1.0, tenure_months)))
    except (ValueError, TypeError):
        total_charges = monthly_charges * max(1.0, tenure_months)

    try:
        cltv = float(get_val("cltv", 4000.0))
    except (ValueError, TypeError):
        cltv = 4000.0

    # Scale numeric values
    tenure_scaled = min(max(tenure_months / SCALER_PARAMS["tenure_max"], 0.0), 1.0)
    monthly_scaled = (monthly_charges - SCALER_PARAMS["monthly_charges_mean"]) / SCALER_PARAMS["monthly_charges_std"]
    total_scaled = (total_charges - SCALER_PARAMS["total_charges_mean"]) / SCALER_PARAMS["total_charges_std"]
    cltv_scaled = (cltv - SCALER_PARAMS["cltv_mean"]) / SCALER_PARAMS["cltv_std"]

    # Categorical multi-level columns
    multiple_lines = str(get_val("multiple lines", "No")).strip()
    internet_service = str(get_val("internet service", "Fiber optic")).strip()
    online_security = str(get_val("online security", "No")).strip()
    online_backup = str(get_val("online backup", "No")).strip()
    device_protection = str(get_val("device protection", "No")).strip()
    tech_support = str(get_val("tech support", "No")).strip()
    streaming_tv = str(get_val("streaming tv", "No")).strip()
    streaming_movies = str(get_val("streaming movies", "No")).strip()
    contract = str(get_val("contract", "Month-to-month")).strip()
    payment_method = str(get_val("payment method", "Electronic check")).strip()

    row = {
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure_scaled,
        "Phone Service": phone_service,
        "Paperless Billing": paperless_billing,
        "Monthly Charges": monthly_scaled,
        "Total Charges": total_scaled,
        "CLTV": cltv_scaled,
        "Multiple Lines_No phone service": 1 if multiple_lines.lower() == "no phone service" else 0,
        "Multiple Lines_Yes": 1 if multiple_lines.lower() == "yes" else 0,
        "Internet Service_Fiber optic": 1 if internet_service.lower() == "fiber optic" else 0,
        "Internet Service_No": 1 if internet_service.lower() == "no" else 0,
        "Online Security_No internet service": 1 if online_security.lower() == "no internet service" else 0,
        "Online Security_Yes": 1 if online_security.lower() == "yes" else 0,
        "Online Backup_No internet service": 1 if online_backup.lower() == "no internet service" else 0,
        "Online Backup_Yes": 1 if online_backup.lower() == "yes" else 0,
        "Device Protection_No internet service": 1 if device_protection.lower() == "no internet service" else 0,
        "Device Protection_Yes": 1 if device_protection.lower() == "yes" else 0,
        "Tech Support_No internet service": 1 if tech_support.lower() == "no internet service" else 0,
        "Tech Support_Yes": 1 if tech_support.lower() == "yes" else 0,
        "Streaming TV_No internet service": 1 if streaming_tv.lower() == "no internet service" else 0,
        "Streaming TV_Yes": 1 if streaming_tv.lower() == "yes" else 0,
        "Streaming Movies_No internet service": 1 if streaming_movies.lower() == "no internet service" else 0,
        "Streaming Movies_Yes": 1 if streaming_movies.lower() == "yes" else 0,
        "Contract_One year": 1 if contract.lower() == "one year" else 0,
        "Contract_Two year": 1 if contract.lower() == "two year" else 0,
        "Payment Method_Credit card (automatic)": 1 if "credit card" in payment_method.lower() else 0,
        "Payment Method_Electronic check": 1 if "electronic check" in payment_method.lower() else 0,
        "Payment Method_Mailed check": 1 if "mailed check" in payment_method.lower() else 0,
    }

    df = pd.DataFrame([row], columns=MODEL_FEATURE_NAMES)
    return df


def extract_risk_factors(customer: Dict[str, Any], prob: float) -> List[str]:
    """Identify top contributing churn risk factors for business explainability."""
    factors = []
    norm = {_normalize_key(k): v for k, v in customer.items()}

    contract = str(norm.get("contract", "")).strip().lower()
    if "month-to-month" in contract:
        factors.append("Month-to-month contract (Highest churn segment: 45.3%)")

    internet = str(norm.get("internet service", "")).strip().lower()
    security = str(norm.get("online security", "")).strip().lower()
    tech_support = str(norm.get("tech support", "")).strip().lower()
    if "fiber" in internet and (security != "yes" or tech_support != "yes"):
        factors.append("Fiber optic subscription without security or tech support bundle")

    payment = str(norm.get("payment method", "")).strip().lower()
    if "electronic check" in payment:
        factors.append("Electronic check payment (Lacks automatic recurring payment)")

    try:
        tenure = float(norm.get("tenure months", 24))
        if tenure <= 6:
            factors.append(f"Critical early tenure period ({int(tenure)} months)")
        elif tenure <= 12:
            factors.append(f"Short customer tenure ({int(tenure)} months)")
    except (ValueError, TypeError):
        pass

    try:
        monthly = float(norm.get("monthly charges", 50))
        if monthly >= 80:
            factors.append(f"High monthly bill (${monthly:.2f}/mo)")
    except (ValueError, TypeError):
        pass

    if not factors and prob > 0.5:
        factors.append("Combined behavioral risk indicators")
    elif not factors:
        factors.append("Low risk profile: long-term contract and service stability")

    return factors


def predict_churn(customer: Dict[str, Any], model=None) -> Dict[str, Any]:
    """Predict churn probability and class for a single customer record."""
    if model is None:
        model = load_model()

    df_features = preprocess_single_record(customer)
    prediction = int(model.predict(df_features)[0])
    prob = float(model.predict_proba(df_features)[0][1])

    risk_level = "High" if prob >= 0.65 else "Medium" if prob >= 0.35 else "Low"
    risk_factors = extract_risk_factors(customer, prob)

    return {
        "churn_prediction": prediction,
        "churn_label": "Churn" if prediction == 1 else "Retained",
        "churn_probability": round(prob, 4),
        "risk_level": risk_level,
        "top_risk_factors": risk_factors,
        "model_version": "Tuned XGBoost (Optuna)",
    }


def predict_batch(customers: List[Dict[str, Any]], model=None) -> List[Dict[str, Any]]:
    """Predict churn for a batch of customer records."""
    if model is None:
        model = load_model()

    dfs = [preprocess_single_record(c) for c in customers]
    df_batch = pd.concat(dfs, ignore_index=True)

    predictions = model.predict(df_batch)
    probabilities = model.predict_proba(df_batch)[:, 1]

    results = []
    for c, pred, prob in zip(customers, predictions, probabilities):
        prob_float = float(prob)
        risk_level = "High" if prob_float >= 0.65 else "Medium" if prob_float >= 0.35 else "Low"
        results.append({
            "churn_prediction": int(pred),
            "churn_label": "Churn" if int(pred) == 1 else "Retained",
            "churn_probability": round(prob_float, 4),
            "risk_level": risk_level,
            "top_risk_factors": extract_risk_factors(c, prob_float),
            "model_version": "Tuned XGBoost (Optuna)",
        })

    return results