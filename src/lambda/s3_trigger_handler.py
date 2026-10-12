import json
import logging
import os
import boto3
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
sns_client = boto3.client('sns')

SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', '')
VALID_EXTENSIONS = ('.xlsx', '.csv', '.parquet')


def lambda_handler(event, context):
    """Triggered by S3 PUT event on the /raw/ prefix."""
    logger.info(f"Event received: {json.dumps(event)}")

    for record in event.get('Records', []):
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        size = record['s3']['object'].get('size', 0)
        timestamp = datetime.utcnow().isoformat()

        logger.info(f"[{timestamp}] New file detected: s3://{bucket}/{key} ({size} bytes)")

        # 1. Validate file extension
        if not key.lower().endswith(VALID_EXTENSIONS):
            reason = f"Unsupported file format. Expected: {VALID_EXTENSIONS}"
            logger.warning(f"REJECTED: {key} — {reason}")
            _move_to_rejected(bucket, key, reason)
            _send_alert(bucket, key, reason)
            return _response(400, f"Rejected: {reason}")

        # 2. Validate file size (reject if empty or > 50MB)
        if size == 0:
            reason = "File is empty (0 bytes)"
            logger.warning(f"REJECTED: {key} — {reason}")
            _move_to_rejected(bucket, key, reason)
            _send_alert(bucket, key, reason)
            return _response(400, f"Rejected: {reason}")

        if size > 50 * 1024 * 1024:  # 50MB
            reason = f"File exceeds 50MB limit ({size} bytes)"
            logger.warning(f"REJECTED: {key} — {reason}")
            _move_to_rejected(bucket, key, reason)
            _send_alert(bucket, key, reason)
            return _response(400, f"Rejected: {reason}")

        # 3. File passes basic validation — log success
        logger.info(f"ACCEPTED: {key} — basic validation passed, ready for data validation")

        # In a full pipeline, you would trigger the GX validation here
        # For this project, GX validation runs locally as part of the
        # preprocessing pipeline (see src/churn_predictor/validation.py)

    return _response(200, "File accepted for processing")


def _move_to_rejected(bucket, key, reason):
    """Copy file to rejected/ prefix and delete from raw/."""
    filename = key.split('/')[-1]
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    rejected_key = f"rejected/{timestamp}_{filename}"

    try:
        s3_client.copy_object(
            Bucket=bucket,
            CopySource={'Bucket': bucket, 'Key': key},
            Key=rejected_key,
            Metadata={'rejection_reason': reason},
            MetadataDirective='REPLACE'
        )
        s3_client.delete_object(Bucket=bucket, Key=key)
        logger.info(f"Moved {key} → {rejected_key}")
    except Exception as e:
        logger.error(f"Failed to move file to rejected: {e}")


def _send_alert(bucket, key, reason):
    """Send SNS notification about rejected file."""
    if not SNS_TOPIC_ARN:
        logger.warning("SNS_TOPIC_ARN not configured, skipping alert")
        return

    try:
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject="[Churn Predictor] Data Validation Alert",
            Message=(
                f"File Rejected\n"
                f"Bucket: {bucket}\n"
                f"File: {key}\n"
                f"Reason: {reason}\n"
                f"Time: {datetime.utcnow().isoformat()}\n"
                f"\nAction Required: Review the file and re-upload a corrected version."
            )
        )
        logger.info("SNS alert sent successfully")
    except Exception as e:
        logger.error(f"Failed to send SNS alert: {e}")


def _response(status_code, message):
    return {'statusCode': status_code, 'body': json.dumps({'message': message})}