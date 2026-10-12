"""Flask API for churn prediction."""
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from churn_predictor.predict import load_model, predict_churn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load model at startup
model = None


@app.before_request
def load_model_once():
    global model
    if model is None:
        logger.info("Loading model...")
        model = load_model("best_churn_model")
        logger.info("Model loaded successfully")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """Predict churn for a customer."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        result = predict_churn(data, model=model)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Predict churn for multiple customers."""
    try:
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({"error": "Expected a JSON array"}), 400

        results = [predict_churn(customer, model=model) for customer in data]
        return jsonify({"predictions": results, "count": len(results)})

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)