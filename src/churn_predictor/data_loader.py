"""Load data from and save data to AWS S3."""
import logging
import boto3
from pathlib import Path
from io import BytesIO
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def get_s3_client():
    """Create an S3 client using default credential chain."""
    return boto3.client('s3', region_name='ap-southeast-2')


def upload_file(local_path: Path, s3_key: str, bucket: str = None):
    """Upload a local file to S3."""
    bucket = bucket or config.S3_BUCKET
    s3 = get_s3_client()
    s3.upload_file(str(local_path), bucket, s3_key)
    logger.info(f"Uploaded {local_path} → s3://{bucket}/{s3_key}")


def download_file(s3_key: str, local_path: Path, bucket: str = None):
    """Download a file from S3 to local."""
    bucket = bucket or config.S3_BUCKET
    s3 = get_s3_client()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, s3_key, str(local_path))
    logger.info(f"Downloaded s3://{bucket}/{s3_key} → {local_path}")


def upload_processed_data():
    """Upload all processed CSV files to S3."""
    for filename in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        local_path = config.PROCESSED_DIR / filename
        if local_path.exists():
            upload_file(local_path, f"processed/{filename}")


def upload_model(model_path: Path):
    """Upload a trained model artifact to S3."""
    upload_file(model_path, f"output/models/{model_path.name}")


def upload_report(report_path: Path):
    """Upload a report artifact to S3."""
    upload_file(report_path, f"output/reports/{report_path.name}")