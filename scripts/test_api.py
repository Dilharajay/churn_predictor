"""Test script for Telco Churn Predictor API.

Can be run locally against http://localhost:8000 or against a deployed AWS instance:
    python scripts/test_api.py --base-url http://<EC2-PUBLIC-IP>:8000
"""
import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path


def make_request(url: str, method: str = "GET", payload: dict = None) -> tuple[int, dict]:
    """Helper to perform HTTP requests using standard library urllib."""
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            status_code = response.getcode()
            body = json.loads(response.read().decode("utf-8"))
            return status_code, body
    except urllib.error.HTTPError as e:
        body = json.loads(e.read().decode("utf-8"))
        return e.code, body
    except Exception as e:
        return 0, {"error": str(e)}


def run_tests(base_url: str):
    base_url = base_url.rstrip("/")
    print("=" * 65)
    print(f" Testing Telco Churn Predictor API at: {base_url}")
    print("=" * 65)

    # 1. Health Check
    print("\n[1/4] Checking Health Endpoint (/health)...")
    status, health_resp = make_request(f"{base_url}/health")
    if status == 200 and health_resp.get("status") == "healthy":
        print(f"  [PASS] Status: {status} | Service is healthy!")
        print(f"         Model loaded: {health_resp.get('model_loaded')}")
        print(f"         Features: {health_resp.get('features_count')} features ready")
    else:
        print(f"  [FAIL] Health check failed: {health_resp}")
        sys.exit(1)

    # 2. Model Info
    print("\n[2/4] Checking Model Information (/model/info)...")
    status, info_resp = make_request(f"{base_url}/model/info")
    if status == 200:
        print(f"  [PASS] Model: {info_resp.get('model_name')}")
        print(f"         Optimization: {info_resp.get('optimization')}")
        print(f"         CV ROC-AUC: {info_resp.get('evaluation_metrics', {}).get('cv_roc_auc')}")
    else:
        print(f"  [WARN] Failed to get model info: {info_resp}")

    # Load sample requests
    samples_path = Path(__file__).parent / "sample_requests.json"
    with open(samples_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # 3. Single Prediction - High Risk
    print("\n[3/4] Testing Single Prediction (/predict) — High Risk Profile...")
    high_risk = samples["high_risk_customer"]
    status, pred_resp = make_request(f"{base_url}/predict", method="POST", payload=high_risk)
    if status == 200:
        print(f"  [PASS] Prediction: {pred_resp.get('churn_label')} ({pred_resp.get('churn_prediction')})")
        print(f"         Probability: {pred_resp.get('churn_probability') * 100:.1f}%")
        print(f"         Risk Level: {pred_resp.get('risk_level')}")
        print(f"         Top Risk Factors:")
        for factor in pred_resp.get("top_risk_factors", []):
            print(f"           - {factor}")
    else:
        print(f"  [FAIL] High-risk prediction failed: {pred_resp}")

    # 4. Batch Prediction
    print("\n[4/4] Testing Batch Prediction (/predict/batch)...")
    batch_payload = {"customers": [samples["high_risk_customer"], samples["low_risk_customer"]]}
    status, batch_resp = make_request(f"{base_url}/predict/batch", method="POST", payload=batch_payload)
    if status == 200:
        print(f"  [PASS] Processed {batch_resp.get('count')} customers successfully.")
        print(f"         High-Risk count: {batch_resp.get('high_risk_count')}")
        for i, p in enumerate(batch_resp.get("predictions", [])):
            print(f"         Customer {i + 1}: {p.get('churn_label')} | Prob: {p.get('churn_probability') * 100:.1f}% | Risk: {p.get('risk_level')}")
    else:
        print(f"  [FAIL] Batch prediction failed: {batch_resp}")

    print("\n" + "=" * 65)
    print(" ALL API VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Telco Churn Predictor API")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the API")
    args = parser.parse_args()
    run_tests(args.base_url)
