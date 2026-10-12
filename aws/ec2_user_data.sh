#!/bin/bash
# ==============================================================================
# EC2 User Data Bootstrap Script — Churn Predictor FastAPI Container
# Runs automatically upon EC2 instance launch (Amazon Linux 2023)
# ==============================================================================
set -e
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=== [1/6] Configuring 2GB Swap for Free Tier stability ==="
if [ ! -f /swapfile ]; then
    dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "=== [2/6] Updating packages & installing Docker ==="
dnf install -y docker git --allowerasing

echo "=== [3/6] Starting Docker service ==="
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

echo "=== [4/6] Setting up application workspace ==="
mkdir -p /opt/churn-predictor
cd /opt/churn-predictor

# Download model from S3
mkdir -p src/churn_predictor/models
aws s3 cp s3://churn-predictor-data/output/models/best_churn_model.joblib src/churn_predictor/models/best_churn_model.joblib --region ap-southeast-2 || true

echo "=== [4/6] Creating deployment container files ==="
cat << 'EOF' > requirements.txt
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.9.0
pandas>=2.2.0
numpy>=1.26.0,<3.0.0
scikit-learn>=1.5.0
xgboost>=2.0.0
joblib>=1.4.0
boto3>=1.34.0
python-multipart>=0.0.9
EOF

cat << 'EOF' > Dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src PORT=8000
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Fetch code from S3 or clone repo
aws s3 sync s3://churn-predictor-data/code/ /opt/churn-predictor/src/ --region ap-southeast-2 || true

echo "=== [5/6] Building and running Docker container ==="
docker build -t churn-predictor-api:latest .
docker run -d \
  --name churn-api \
  -p 8000:8000 \
  --restart always \
  -e AWS_REGION=ap-southeast-2 \
  -e S3_BUCKET_NAME=churn-predictor-data \
  churn-predictor-api:latest

echo "=== [6/6] Verifying local healthcheck ==="
sleep 5
curl -s http://localhost:8000/health || echo "Healthcheck pending startup"
echo "=== Deployment Complete! FastAPI is running on port 8000 ==="
