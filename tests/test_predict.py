"""Unit tests for the churn prediction inference engine."""
import pytest
from churn_predictor.predict import (
    load_model,
    predict_churn,
    predict_batch,
    MODEL_FEATURE_NAMES,
)


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture
def sample_high_risk():
    return {
        "gender": "Female",
        "senior_citizen": "Yes",
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
        "monthly_charges": 85.50,
        "total_charges": 171.00,
        "cltv": 2700.0,
    }


@pytest.fixture
def sample_low_risk():
    return {
        "gender": "Male",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "Yes",
        "tenure_months": 60.0,
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Two year",
        "paperless_billing": "No",
        "payment_method": "Credit card (automatic)",
        "monthly_charges": 64.00,
        "total_charges": 3840.00,
        "cltv": 5800.0,
    }


def test_model_loading(model):
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_predict_churn_high_risk(model, sample_high_risk):
    result = predict_churn(sample_high_risk, model=model)
    assert "churn_prediction" in result
    assert "churn_probability" in result
    assert "risk_level" in result
    assert "top_risk_factors" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["churn_prediction"] in (0, 1)
    assert result["risk_level"] in ("Low", "Medium", "High")
    # High-risk profile expectation
    assert result["churn_probability"] > 0.5
    assert result["risk_level"] in ("Medium", "High")


def test_predict_churn_low_risk(model, sample_low_risk):
    result = predict_churn(sample_low_risk, model=model)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["churn_prediction"] == 0
    assert result["risk_level"] == "Low"


def test_predict_batch(model, sample_high_risk, sample_low_risk):
    batch = [sample_high_risk, sample_low_risk]
    results = predict_batch(batch, model=model)
    assert len(results) == 2
    assert results[0]["churn_prediction"] == 1
    assert results[1]["churn_prediction"] == 0
