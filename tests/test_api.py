"""Integration tests for the FastAPI churn prediction endpoints."""
import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "ChurnPredictor" in response.text


def test_api_info_endpoint(client):
    response = client.get("/api")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["features_count"] == 31


def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "evaluation_metrics" in data


def test_predict_endpoint(client):
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "top_risk_factors" in data


def test_predict_batch_endpoint(client):
    payload = {
        "customers": [
            {
                "gender": "Female",
                "tenure_months": 2.0,
                "contract": "Month-to-month",
                "monthly_charges": 85.50,
            },
            {
                "gender": "Male",
                "tenure_months": 60.0,
                "contract": "Two year",
                "monthly_charges": 64.00,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert "predictions" in data
    assert len(data["predictions"]) == 2
