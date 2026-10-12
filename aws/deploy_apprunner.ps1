<#
.SYNOPSIS
    Automated deployment script for Telco Churn Predictor FastAPI on AWS App Runner via ECR (ap-southeast-2).
.DESCRIPTION
    1. Creates Amazon ECR Repository 'churn-predictor-api'.
    2. Logs into Amazon ECR.
    3. Builds the Docker container image.
    4. Pushes the image to Amazon ECR.
    5. Creates/Updates the AWS App Runner Service with automatic HTTPS URL.
#>
param (
    [string]$Region = "ap-southeast-2",
    [string]$AccountId = "",
    [string]$RepoName = "churn-predictor-api",
    [string]$ServiceName = "churn-predictor-api-service"
)

$ErrorActionPreference = "Stop"
$aws = "C:\Users\Dilhara Jayawardhana\AppData\Local\Programs\Amazon\AWSCLIV2\aws.exe"
if (-not (Test-Path $aws)) { $aws = "aws" }

if (-not $AccountId) {
    $AccountId = (& $aws sts get-caller-identity --query "Account" --output text)
}

$ecrRegistry = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$imageUri = "$ecrRegistry/${RepoName}:latest"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Deploying Telco Churn Predictor API to AWS App Runner" -ForegroundColor Cyan
Write-Host " ECR Image: $imageUri" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Create ECR repository if it does not exist
Write-Host "[1/5] Checking/Creating ECR Repository '$RepoName'..." -ForegroundColor Yellow
$repoExists = (& $aws ecr describe-repositories --repository-names $RepoName --region $Region 2>$null)
if (-not $repoExists) {
    & $aws ecr create-repository --repository-name $RepoName --image-scanning-configuration scanOnPush=true --region $Region | Out-Null
    Write-Host "Created ECR repository: $RepoName" -ForegroundColor Green
} else {
    Write-Host "Found existing ECR repository: $RepoName" -ForegroundColor Green
}

# 2. Login Docker to ECR
Write-Host "[2/5] Authenticating Docker to Amazon ECR..." -ForegroundColor Yellow
& $aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $ecrRegistry

# 3. Build Docker image
Write-Host "[3/5] Building Docker image..." -ForegroundColor Yellow
docker build -t "${RepoName}:latest" .
docker tag "${RepoName}:latest" $imageUri

# 4. Push to ECR
Write-Host "[4/5] Pushing Docker image to ECR..." -ForegroundColor Yellow
docker push $imageUri

# 5. Create App Runner Access Role if not present
Write-Host "[5/5] Configuring App Runner Service..." -ForegroundColor Yellow
$roleName = "AppRunnerECRAccessRole"
$roleArn = (& $aws iam get-role --role-name $roleName --query "Role.Arn" --output text 2>$null)

if (-not $roleArn -or $roleArn -eq "None") {
    $trustPolicy = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"build.apprunner.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    $roleArn = (& $aws iam create-role --role-name $roleName --assume-role-policy-document $trustPolicy --query "Role.Arn" --output text)
    & $aws iam attach-role-policy --role-name $roleName --policy-arn "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
    Start-Sleep -Seconds 10
}

# Create App Runner service
$serviceArn = (& $aws apprunner create-service `
    --service-name $ServiceName `
    --source-configuration "{
        \"AuthenticationConfiguration\": {\"AccessRoleArn\": \"$roleArn\"},
        \"AutoDeploymentsEnabled\": true,
        \"ImageRepository\": {
            \"ImageIdentifier\": \"$imageUri\",
            \"ImageConfiguration\": {
                \"Port\": \"8000\",
                \"RuntimeEnvironmentVariables\": {
                    \"AWS_REGION\": \"$Region\",
                    \"S3_BUCKET_NAME\": \"churn-predictor-data\"
                }
            },
            \"ImageRepositoryType\": \"ECR\"
        }
    }" `
    --region $Region `
    --query "Service.ServiceArn" `
    --output text)

Write-Host "App Runner Service initialized: $serviceArn" -ForegroundColor Green
Write-Host "Waiting for service deployment..." -ForegroundColor Yellow

$serviceUrl = (& $aws apprunner describe-service --service-arn $serviceArn --region $Region --query "Service.ServiceUrl" --output text)

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host " APP RUNNER DEPLOYMENT ACTIVE!" -ForegroundColor Green
Write-Host " Service URL:   https://$serviceUrl" -ForegroundColor Cyan
Write-Host " Swagger Docs:  https://$serviceUrl/docs" -ForegroundColor Cyan
Write-Host " Health Check:  https://$serviceUrl/health" -ForegroundColor Cyan
Write-Host " Predict URL:   https://$serviceUrl/predict" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green
