"""Central configuration for the churn prediction pipeline."""
import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REJECTED_DIR = DATA_DIR / "rejected"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- S3 & AWS ---
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "churn-predictor-data")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID", "")

# --- Dataset ---
RAW_FILENAME = "Telco_customer_churn.xlsx"
TARGET = "Churn Value"

# --- Columns to drop ---
DROP_COLUMNS = [
    "CustomerID", "Count", "Country", "State", "City",
    "Zip Code", "Lat Long", "Latitude", "Longitude",
    "Churn Label", "Churn Score", "Churn Reason",
]

# --- Feature groups ---
NUMERIC_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]

BINARY_CATEGORICAL = [
    "Gender", "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Paperless Billing",
]

MULTI_CATEGORICAL = [
    "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies", "Contract", "Payment Method",
]

# --- Scaling ---
STANDARD_SCALED = ["Monthly Charges", "Total Charges", "CLTV"]
MINMAX_SCALED = ["Tenure Months"]

# --- Model ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

MLFLOW_TRACKING_URI="sqlite:///mlflow.db"