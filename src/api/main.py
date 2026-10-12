"""FastAPI Application for Customer Churn Prediction.

Exposes RESTful endpoints for real-time and batch machine learning inference,
comprehensive input validation with Pydantic, Swagger documentation, and healthchecks.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from churn_predictor.predict import (
    MODEL_FEATURE_NAMES,
    load_model,
    predict_batch,
    predict_churn,
)

logger = logging.getLogger("churn_predictor.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

START_TIME = time.time()
_LOADED_MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the model at startup and clean up on shutdown."""
    global _LOADED_MODEL
    logger.info("Initializing Telco Churn Predictor API...")
    try:
        _LOADED_MODEL = load_model()
        logger.info("Model warmed up successfully.")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
    yield
    logger.info("Shutting down Telco Churn Predictor API...")


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="""
Cloud-Native Machine Learning REST API for predicting telecommunications customer churn.
Powered by a Bayesian-tuned XGBoost classifier (Optuna) evaluated with 5-Fold Stratified Cross-Validation on AWS.

### Features
- **Real-Time Inference (`/predict`)**: Instant single-customer risk scoring & probability.
- **Batch Processing (`/predict/batch`)**: Bulk predictions for CRM / marketing campaigns.
- **Decision Support**: Actionable risk factor explanations alongside predictions.
- **Automated Documentation**: Interactive Swagger docs at `/docs` and ReDoc at `/redoc`.
""",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerInput(BaseModel):
    """Customer features input schema with validation and examples."""

    gender: Optional[str] = Field("Female", description="Gender ('Male' or 'Female')")
    senior_citizen: Optional[str] = Field("No", description="Senior citizen indicator ('Yes' or 'No')")
    partner: Optional[str] = Field("No", description="Has a partner ('Yes' or 'No')")
    dependents: Optional[str] = Field("No", description="Has dependents ('Yes' or 'No')")
    tenure_months: float = Field(2.0, ge=0.0, le=120.0, description="Tenure with company in months")
    phone_service: Optional[str] = Field("Yes", description="Has phone service ('Yes' or 'No')")
    multiple_lines: Optional[str] = Field("No", description="Multiple phone lines ('Yes', 'No', 'No phone service')")
    internet_service: Optional[str] = Field("Fiber optic", description="Internet connection type ('DSL', 'Fiber optic', 'No')")
    online_security: Optional[str] = Field("No", description="Online security add-on ('Yes', 'No', 'No internet service')")
    online_backup: Optional[str] = Field("No", description="Online backup add-on ('Yes', 'No', 'No internet service')")
    device_protection: Optional[str] = Field("No", description="Device protection add-on ('Yes', 'No', 'No internet service')")
    tech_support: Optional[str] = Field("No", description="Tech support add-on ('Yes', 'No', 'No internet service')")
    streaming_tv: Optional[str] = Field("Yes", description="Streaming TV add-on ('Yes', 'No', 'No internet service')")
    streaming_movies: Optional[str] = Field("No", description="Streaming Movies add-on ('Yes', 'No', 'No internet service')")
    contract: Optional[str] = Field("Month-to-month", description="Contract term ('Month-to-month', 'One year', 'Two year')")
    paperless_billing: Optional[str] = Field("Yes", description="Paperless billing opted ('Yes' or 'No')")
    payment_method: Optional[str] = Field("Electronic check", description="Billing method ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')")
    monthly_charges: float = Field(70.70, ge=0.0, le=500.0, description="Monthly recurring charge in USD")
    total_charges: Optional[float] = Field(None, ge=0.0, description="Total lifetime spend in USD (defaults to monthly * tenure)")
    cltv: Optional[float] = Field(3500.0, ge=0.0, le=10000.0, description="Customer Lifetime Value score")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "senior_citizen": "No",
                "partner": "No",
                "dependents": "No",
                "tenure_months": 2.0,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "No",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "No",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 70.70,
                "total_charges": 141.40,
                "cltv": 2701.0,
            }
        }


class PredictionResponse(BaseModel):
    """Prediction outcome structure."""

    churn_prediction: int = Field(..., description="Binary classification (1 = Churn, 0 = Retained)")
    churn_label: str = Field(..., description="Human-readable label ('Churn' or 'Retained')")
    churn_probability: float = Field(..., description="Probability of churn between 0.0 and 1.0")
    risk_level: str = Field(..., description="Risk category: 'Low', 'Medium', or 'High'")
    top_risk_factors: List[str] = Field(..., description="Identified key drivers contributing to churn risk")
    model_version: str = Field(..., description="Name and version of model applied")


class BatchCustomerInput(BaseModel):
    """Batch input schema containing a list of customer records."""

    customers: List[CustomerInput] = Field(..., min_length=1, description="List of customer records to evaluate")


class BatchPredictionResponse(BaseModel):
    """Batch prediction output structure."""

    count: int = Field(..., description="Number of customers evaluated")
    high_risk_count: int = Field(..., description="Number of customers classified as High Risk")
    predictions: List[PredictionResponse] = Field(..., description="List of individual predictions")


class HealthResponse(BaseModel):
    """Service health status and diagnostics."""

    status: str
    model_loaded: bool
    features_count: int
    uptime_seconds: float


@app.get("/", response_class=HTMLResponse, tags=["Dashboard & UI"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard & UI"])
def serve_dashboard():
    """Serves the interactive Customer Churn Prediction web UI."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>Telco Customer Churn Prediction API</h1><p>Visit <a href='/docs'>/docs</a> for interactive Swagger documentation.</p>"
    )


@app.get("/api", tags=["Health & Metadata"])
def api_info():
    """API overview and navigation links for programmatic consumers."""
    return {
        "service": "Telco Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "online",
        "ui_dashboard": "/",
        "documentation": "/docs",
        "health": "/health",
        "model_info": "/model/info",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health & Metadata"])
def health_check():
    """Health check endpoint for Docker / AWS container orchestration."""
    global _LOADED_MODEL
    is_loaded = _LOADED_MODEL is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        features_count=len(MODEL_FEATURE_NAMES),
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


@app.get("/model/info", tags=["Health & Metadata"])
def model_information():
    """Returns architecture and performance metadata of the deployed model."""
    return {
        "model_name": "Tuned XGBoost Classifier",
        "optimization": "Optuna Bayesian Optimization (15 trials)",
        "cross_validation": "5-Fold Stratified Cross-Validation",
        "evaluation_metrics": {
            "cv_roc_auc": 0.8579,
            "cv_recall": 0.8048,
            "test_accuracy": 0.7835,
            "test_f1_score": 0.6258,
        },
        "total_features": len(MODEL_FEATURE_NAMES),
        "feature_list": MODEL_FEATURE_NAMES,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(customer: CustomerInput):
    """
    Score a single customer for churn risk.

    Returns churn binary prediction, churn probability (0-1),
    risk classification, and top driving risk factors.
    """
    global _LOADED_MODEL
    try:
        if _LOADED_MODEL is None:
            _LOADED_MODEL = load_model()

        customer_dict = customer.model_dump()
        result = predict_churn(customer_dict, model=_LOADED_MODEL)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Inference error on single customer: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_multiple(batch: BatchCustomerInput):
    """
    Score a batch of customers in high-throughput vectorized mode.

    Ideal for automated nightly batch scoring or CRM campaign exports.
    """
    global _LOADED_MODEL
    try:
        if _LOADED_MODEL is None:
            _LOADED_MODEL = load_model()

        customers_list = [c.model_dump() for c in batch.customers]
        results = predict_batch(customers_list, model=_LOADED_MODEL)
        high_risk_count = sum(1 for r in results if r["risk_level"] == "High")

        return BatchPredictionResponse(
            count=len(results),
            high_risk_count=high_risk_count,
            predictions=[PredictionResponse(**r) for r in results],
        )
    except Exception as e:
        logger.error(f"Inference error on batch request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}",
        )
