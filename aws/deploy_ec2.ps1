<#
.SYNOPSIS
    Automated deployment script for Telco Churn Predictor FastAPI on AWS EC2 Free Tier (ap-southeast-2).
.DESCRIPTION
    1. Discovers default VPC and Subnet.
    2. Creates Security Group (churn-predictor-api-sg) opening port 8000 and 22.
    3. Finds latest Amazon Linux 2023 AMI.
    4. Launches a t2.micro or t3.micro Free Tier instance with ec2_user_data.sh.
    5. Displays the public IP and live Swagger Documentation URL.
#>
param (
    [string]$Region = "ap-southeast-2",
    [string]$InstanceType = "t2.micro",
    [string]$KeyName = ""  # Optional: specify existing EC2 Key Pair if you want SSH access
)

$ErrorActionPreference = "Stop"
$aws = "C:\Users\Dilhara Jayawardhana\AppData\Local\Programs\Amazon\AWSCLIV2\aws.exe"
if (-not (Test-Path $aws)) { $aws = "aws" }

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Deploying Telco Churn Predictor API to AWS EC2 (Free Tier)" -ForegroundColor Cyan
Write-Host " Region: $Region | Instance: $InstanceType" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Sync latest application source code to S3
Write-Host "[1/5] Syncing latest source code to S3 bucket..." -ForegroundColor Yellow
& $aws s3 sync src/ s3://churn-predictor-data/code/ --exclude "*.pyc" --exclude "*__pycache__*" --exclude "*.zip" --region $Region

# 2. Get Default VPC
Write-Host "[2/5] Finding Default VPC in $Region..." -ForegroundColor Yellow
$vpcId = (& $aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --region $Region --query "Vpcs[0].VpcId" --output text)
if (-not $vpcId -or $vpcId -eq "None") {
    $vpcId = (& $aws ec2 describe-vpcs --region $Region --query "Vpcs[0].VpcId" --output text)
}
Write-Host "Using VPC: $vpcId" -ForegroundColor Green

# 3. Create or reuse Security Group
$sgName = "churn-predictor-api-sg"
Write-Host "[3/5] Configuring Security Group '$sgName'..." -ForegroundColor Yellow
$sgId = (& $aws ec2 describe-security-groups --filters "Name=group-name,Values=$sgName" "Name=vpc-id,Values=$vpcId" --region $Region --query "SecurityGroups[0].GroupId" --output text)

if (-not $sgId -or $sgId -eq "None") {
    $sgId = (& $aws ec2 create-security-group --group-name $sgName --description "Security group for Telco Churn Predictor API" --vpc-id $vpcId --region $Region --query "GroupId" --output text)
    Write-Host "Created Security Group: $sgId" -ForegroundColor Green
    
    # Authorize Port 8000 (FastAPI) and Port 22 (SSH)
    & $aws ec2 authorize-security-group-ingress --group-id $sgId --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region $Region | Out-Null
    & $aws ec2 authorize-security-group-ingress --group-id $sgId --protocol tcp --port 22 --cidr 0.0.0.0/0 --region $Region | Out-Null
    Write-Host "Authorized inbound traffic on ports 8000 (FastAPI) and 22 (SSH)" -ForegroundColor Green
} else {
    Write-Host "Found existing Security Group: $sgId" -ForegroundColor Green
}

# 4. Find latest Amazon Linux 2023 AMI
Write-Host "[4/5] Resolving latest Amazon Linux 2023 AMI ID..." -ForegroundColor Yellow
$amiId = (& $aws ec2 describe-images --owners amazon --filters "Name=name,Values=al2023-ami-2023.*-x86_64" "Name=state,Values=available" --region $Region --query "reverse(sort_by(Images, &CreationDate))[0].ImageId" --output text)
Write-Host "Resolved AMI: $amiId" -ForegroundColor Green

# 5. Launch EC2 Instance with User Data
Write-Host "[5/5] Launching EC2 instance..." -ForegroundColor Yellow
$userDataPath = "aws/ec2_user_data.sh"

$launchArgs = @(
    "ec2", "run-instances",
    "--image-id", $amiId,
    "--count", "1",
    "--instance-type", $InstanceType,
    "--security-group-ids", $sgId,
    "--iam-instance-profile", "Name=churn-ec2-worker-role-v2",
    "--user-data", "file://$userDataPath",
    "--tag-specifications", "ResourceType=instance,Tags=[{Key=Name,Value=ChurnPredictorFastAPI},{Key=Project,Value=ChurnPredictor}]",
    "--region", $Region,
    "--query", "Instances[0].InstanceId",
    "--output", "text"
)

if ($KeyName) {
    $launchArgs += @("--key-name", $KeyName)
}

$instanceId = (& $aws @launchArgs)
Write-Host "Successfully launched EC2 instance: $instanceId" -ForegroundColor Green

Write-Host "Waiting for instance to be in running state and obtain Public IP..." -ForegroundColor Yellow
& $aws ec2 wait instance-running --instance-ids $instanceId --region $Region

$publicIp = (& $aws ec2 describe-instances --instance-ids $instanceId --region $Region --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host " DEPLOYMENT SUCCEEDED!" -ForegroundColor Green
Write-Host " Instance ID:   $instanceId" -ForegroundColor Green
Write-Host " Public IP:     $publicIp" -ForegroundColor Green
Write-Host " FastAPI Docs:  http://${publicIp}:8000/docs" -ForegroundColor Cyan
Write-Host " Health Check:  http://${publicIp}:8000/health" -ForegroundColor Cyan
Write-Host " Predict URL:   http://${publicIp}:8000/predict" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Note: It takes ~2-3 minutes for user-data bootstrap (Docker install & build) to complete." -ForegroundColor Yellow
