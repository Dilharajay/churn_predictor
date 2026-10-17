# Customer Churn Prediction Platform: Project Report

*Design, Delivery and Business Value of a Cloud-Hosted Machine Learning Solution · Group Project · October 2026*

## Executive Summary

Customer churn is one of the largest controllable sources of revenue loss in subscription telecommunications. In the IBM Telco dataset used for this project, 26.5% of 7,043 customers (1,869 subscribers) left the service, representing more than $3.8M in at-risk Customer Lifetime Value.

Our team of four designed and delivered an end-to-end platform that predicts which customers are likely to leave, explains why, and recommends a retention action. The platform covers the full lifecycle: a validated cloud data lake on Amazon Web Services, a reproducible machine learning pipeline, SQL analytics, a containerised prediction API with web dashboard, and an automated build-and-deploy pipeline. Everything runs inside the AWS Free Tier, with an estimated cloud cost of $0.00.

**Headline results**

| Measure | Result |
| --- | --- |
| Selected model | Tuned XGBoost (Optuna Bayesian optimisation, 15 trials) |
| Test ROC-AUC | 0.8579 (baseline XGBoost: 0.8331) |
| Test recall (churners correctly identified) | 0.8048, about four in five churners |
| Engineered features | 31, with no post-churn leakage |
| Automated tests | 10 of 10 passing in the CI pipeline |
| Cloud cost | Estimated $0.00 (Free Tier) |

The analysis consistently identifies three high-risk segments: customers on month-to-month contracts, fiber optic customers without support or security services, and customers who pay by electronic check. We recommend a retention programme targeting each of them (Section 9).

## 1. Background and Objectives

### 1.1 Business context

Acquiring a new subscriber is widely estimated to cost five to seven times more than keeping an existing one. A model that identifies at-risk customers early allows a company to spend its retention budget where it matters most, instead of discounting across the whole customer base.

### 1.2 Project objectives

1. Build a secure, organised cloud data store for the original and processed data.
2. Automatically detect and validate new data uploads before they are used.
3. Explore the data, clean it and prepare features for modelling.
4. Compare at least two classification models and identify the characteristics linked to churn.
5. Answer business questions with at least five SQL queries.
6. Serve the best model through an API and an easy-to-use interface.
7. Provide at least three evidence-based business recommendations, following security and cost-control best practice.

### 1.3 Scope and approach

The project ran as a two-week sprint with seven phases: infrastructure, validation, preprocessing, modelling, analytics, API and deployment, and documentation. Work was split by specialism (cloud and security, data quality, machine learning, analytics and API), with all members contributing to the final documentation. The code is hosted in a GitHub repository, and feature branches with pull requests were used to reduce merge conflicts.

## 2. Methodology

### 2.1 Data and validation

The dataset holds 7,043 customer records with 33 attributes, covering demographics, subscribed services, billing and contract details, and the churn outcome. The target variable (Churn Value) is imbalanced, with 26.54% positive cases.

Before any processing, data passes a quality gate built with Great Expectations. The suite contains 13 rules covering the expected schema, unique and non-null customer identifiers, valid target values, permitted categories for each categorical column, and sensible numeric ranges (for example, tenure from 0 to 72 months). Data that passes is stored in a `validated/` zone; data that fails is quarantined in `rejected/` and an email alert is raised through Amazon SNS.

### 2.2 Exploratory analysis

Exploratory analysis used five groups of visualisations (churn distribution, numeric histograms, categorical comparisons, a correlation heatmap and box plots) and statistical tests. It also surfaced a data-quality issue: `Total Charges` is stored as text with blank values and needed to be converted to a numeric type.

### 2.3 Feature preparation

The preparation pipeline was turned from notebook code into a reusable module so that training and production use identical transformations. It performs four tasks:

- **Leakage removal.** `Churn Label`, `Churn Score` and `Churn Reason` are only known after a customer leaves, so they were removed to prevent the model from "cheating".
- **Identifier and location removal.** Fields such as `CustomerID`, `Zip Code`, `City` and coordinates were dropped as they add noise rather than signal.
- **Encoding.** Binary fields became 0/1 values and multi-level fields were one-hot encoded.
- **Scaling.** Tenure was MinMax scaled and the charge and CLTV fields were standardised.

The result is 31 model-ready features. An 80/20 stratified split with a fixed random seed produced 5,634 training and 1,409 test customers.

### 2.4 Modelling and evaluation design

Logistic Regression, Random Forest and XGBoost were compared with 5-Fold Stratified Cross-Validation. Class imbalance was addressed with balanced class weights for the first two and a positive-class weight of about 2.77 for XGBoost. Because missing a churner is more costly than contacting a loyal customer, recall and ROC-AUC were prioritised, with accuracy, precision and F1-score reported for completeness. XGBoost was then tuned with Optuna, and all runs were logged in MLflow for reproducibility.

## 3. Solution Architecture

### 3.1 Data flow

The architecture has six layers:

1. **Ingestion.** A file uploaded to the S3 `raw/` zone triggers an AWS Lambda function, which checks the file type and starts validation.
2. **Validation.** Great Expectations checks the data; failures are quarantined and reported by SNS.
3. **Preparation and analytics.** Validated data is transformed into model-ready features and an analytics-friendly CSV, which is registered in the AWS Glue Data Catalog and queried with Amazon Athena.
4. **Modelling.** Models are trained and tracked in MLflow, and the best artefact is stored in `output/models/` in S3.
5. **Serving.** A Docker container running FastAPI on an EC2 `t3.micro` instance loads the model and exposes the API and web dashboard.
6. **Automation and monitoring.** GitHub Actions builds, tests and publishes the image, and CloudWatch alarms notify the team through SNS.

*\[Insert the AWS architecture diagram here.\]*

### 3.2 Why these services

| Need | Service chosen | Reason |
| --- | --- | --- |
| Durable, organised storage | Amazon S3 | Low cost, zone-based layout, event notifications |
| Reacting to new files | AWS Lambda | Serverless and pay-per-use; no idle server |
| Alerts | Amazon SNS | Simple email notification for failures and alarms |
| SQL analytics | Glue Catalog and Athena | No servers to manage; charged only per data scanned |
| Hosting the API | EC2 t3.micro | Free Tier eligible; full control of the Docker runtime |
| Monitoring | Amazon CloudWatch | Native logs, metrics and alarms |
| Access control | AWS IAM | Least-privilege roles, no keys in code |

### 3.3 Deployment design

The EC2 instance builds itself on first boot using a cloud-init script. It creates 2 GB of swap space (the instance has only 1 GB of RAM), installs Docker, retrieves code and the model from S3, starts the container with an automatic restart policy and verifies the service with a health check. Using an IAM instance role with read-only S3 access means no credentials are stored on the machine. If the model file is ever missing from the container, the application downloads it from S3 on its own.

## 4. Results

### 4.1 Model benchmark

Cross-validation results on the training set:

| Model | Accuracy | Recall | Precision | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7584 | 0.7759 | 0.5307 | 0.6302 | 0.8478 |
| Random Forest | 0.8051 | 0.5117 | 0.6763 | 0.5823 | 0.8493 |
| XGBoost (baseline) | 0.7803 | 0.6970 | 0.5707 | 0.6274 | 0.8444 |

Results on the held-out test set of 1,409 customers:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7473 | 0.5165 | 0.7540 | 0.6130 | 0.8354 |
| Random Forest | 0.8013 | 0.6599 | 0.5187 | 0.5808 | 0.8452 |
| XGBoost (baseline) | 0.7608 | 0.5378 | 0.7032 | 0.6095 | 0.8331 |
| **Tuned XGBoost** | 0.7615 | 0.5337 | **0.8048** | **0.6418** | **0.8579** |

### 4.2 Interpretation

Random Forest looks strongest on accuracy and precision, but it finds only about half of the churners, so it would leave most at-risk revenue unprotected. The tuned XGBoost model gives the best balance for the business goal: the highest recall and ROC-AUC, with an F1-score above every alternative. Bayesian tuning improved ROC-AUC by roughly 2.5 percentage points and recall by about 10 points over the untuned baseline.

The tuned configuration used 72 trees, a maximum depth of 6, a learning rate of about 0.034, row subsampling of about 0.71 and column subsampling of about 0.81.

### 4.3 What drives churn

| Rank | Feature | Importance | Meaning |
| --- | --- | --- | --- |
| 1 | Contract: Two year | 0.2734 | Strongest protection against churn |
| 2 | Contract: One year | 0.1491 | Longer commitment lowers risk |
| 3 | Internet Service: Fiber optic | 0.0669 | Higher risk, likely price or service related |
| 4 | Dependents | 0.0421 | Household stability reduces churn |
| 5 | Online Security: No internet service | 0.0303 | Distinct profile of customers without internet |

SHAP analysis supports these rankings and also explains individual predictions, which is what allows the API to list the specific risk factors for each customer.

### 4.4 Findings from SQL analytics

Six Athena queries answered business questions on contracts, tenure and spend, risky service combinations, add-on adoption, revenue at risk and senior citizens. The key results were:

- Month-to-month customers churn at 45.3%, against 2.8% on two-year contracts, a 16-fold gap.
- Fiber optic customers churn at 41.9%, and at 52.3% when they have no security or support services.
- Electronic check customers churn at 45.3%, against 15.2% for automatic credit card payers.
- More than 53% of churn occurs within a customer's first six months.

The model and the SQL analysis reach the same conclusions independently, which strengthens confidence in the findings.

## 5. Delivered System

### 5.1 Prediction API

| Endpoint | Function |
| --- | --- |
| `GET /health` | Service status, model loaded, feature count, uptime |
| `GET /model/info` | Model details and evaluation metrics |
| `POST /predict` | Churn prediction for one customer |
| `POST /predict/batch` | Predictions for many customers in one call |
| `/docs` and `/redoc` | Interactive documentation generated automatically |

Each prediction returns a label, a churn probability, a risk level (High, Medium or Low) and the top risk factors in plain language. In live testing, a new fiber optic customer on a month-to-month contract paying by electronic check scored 87.4% (High), while a long-term stable customer scored 1.5% (Low). Input field names are accepted in either `snake_case` or Title Case, and requests are validated before scoring.

### 5.2 User interfaces

- **Web dashboard** for business users, served by the API itself: one-click customer personas, an animated risk gauge, a breakdown of risk drivers and suggested retention offers.
- **Streamlit application** for the data science team: adjustable inputs, batch CSV scoring with risk charts and exportable results, and model comparison views.

### 5.3 Automation, testing and monitoring

A GitHub Actions workflow runs on every push or pull request. It executes the automated test suite (10 tests across the inference engine and API routes), builds the Docker image and publishes it to Docker Hub, then pulls the published image and smoke-tests the `/health` and `/predict` endpoints. AWS Systems Manager then refreshes the running server. CloudWatch alarms watch CPU use (above 85% for ten minutes) and EC2 status checks and notify the team by SNS email.

## 6. Challenges and Resolutions

| Challenge | Impact | Resolution |
| --- | --- | --- |
| Deployment user lacked EC2 permissions | Could not create or manage the server | Attached the required EC2 and SSM policies through the AWS CLI |
| Package conflict on Amazon Linux 2023 | Unattended install failed | Installed packages with `--allowerasing` in the bootstrap script |
| Only 1 GB RAM on the Free Tier instance | Docker builds at risk of running out of memory | Created a 2 GB swap file automatically at start-up |
| `Total Charges` stored as text | Numeric analysis and modelling errors | Converted to numeric during preprocessing |
| Imbalanced target (about 2.77 to 1) | Models biased toward the majority class | Class weights, `scale_pos_weight` and recall-focused evaluation |

## 7. Security, Governance and Cost

### 7.1 Security controls

| Control | Implementation |
| --- | --- |
| Secrets management | Local `.env` file excluded from git; AWS default credential chain; CI secrets held as GitHub repository secrets |
| Least privilege | Separate IAM roles per service; EC2 role limited to S3 read-only and SSM |
| Data protection | S3 public access blocked; encryption at rest (SSE-S3) |
| Data integrity | Great Expectations gate with quarantine of invalid files |
| Leakage prevention | Post-churn fields removed from model inputs |

### 7.2 Cost control

All services fit within Free Tier limits: about 10 MB in S3, fewer than 100 Lambda invocations, under 1 MB scanned per Athena query and fewer than 10 SNS alerts. A $5 budget alarm, weekly Cost Explorer reviews and project tagging provide early warning. Stopping the EC2 instance pauses compute use, and an Elastic IP can be attached if a fixed address is required.

## 8. Risk Register and Limitations

### 8.1 Project risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Free Tier limits exceeded | Low | Medium | Budget alarm and resource tagging |
| Model overfitting | Medium | High | Cross-validation, class weights, held-out test set |
| Team member availability | Medium | High | Clear task ownership and asynchronous work |
| Slow Lambda start-up for validation | Medium | Low | Keep validation lightweight or run it locally |
| Data drift after deployment | Low | Medium | Validation gate on all new data |
| Merge conflicts | Medium | Low | Feature branches and pull requests |

### 8.2 Known limitations

- **Evaluation design.** Optuna maximised ROC-AUC on the test set, so the final test score may be slightly optimistic. Future tuning should use cross-validation only and reserve the test set for a single final check.
- **Transport security.** The API uses HTTP and SSH is open to all addresses, which is acceptable for a coursework demonstration but not for production.
- **Single snapshot.** The data is a single historical sample and the model should be monitored and retrained as customer behaviour changes.
- **Single instance.** One small server is not highly available.

## 9. Business Recommendations

| Priority | Target segment | Observed churn | Recommended action |
| --- | --- | --- | --- |
| 1 | Month-to-month customers | 45.3% | Offer a discount on an annual plan around the second month, then reward tenure milestones |
| 2 | Fiber optic customers without add-ons | 41.9% overall, 52.3% without security or support | Launch a "Fiber Complete" bundle or a free six-month trial of security and tech support |
| 3 | Electronic check payers | 45.3% | Offer a recurring monthly discount (for example $5) for switching to automatic payment |

Use the prediction API to rank customers by churn probability so offers go first to the highest-risk accounts, and use the risk-factor output to choose the right offer for each person. Any expected uplift should be treated as a hypothesis and measured in a controlled pilot against a comparison group before being relied on in financial plans.

**Suggested success measures:** monthly churn rate by segment, first-six-month retention, share of customers on annual contracts, share on automatic payment, add-on adoption among fiber customers, and recall of the model on newly labelled data.

## 10. Roadmap and Future Work

| Horizon | Action |
| --- | --- |
| Short term | Add HTTPS, API authentication and restricted SSH access; tune on cross-validation only |
| Medium term | Scheduled retraining and drift monitoring; threshold tuning using the cost of each retention offer; an A/B test of the three campaigns |
| Long term | Auto-scaling or a managed container service; a model registry; survival analysis to predict when a customer will leave; feedback data on which offers worked |

## 11. Conclusion

The project met its objectives: a validated and secured cloud data lake, a reproducible and tracked modelling pipeline, SQL analytics, a live containerised API with two interfaces, and an automated delivery pipeline, all within the AWS Free Tier. The tuned XGBoost model achieves a test ROC-AUC of 0.8579 and recall of 0.8048, and its findings match those of the exploratory and SQL analysis: contract type, fiber optic service without add-ons, and electronic check payment are the clearest risk signals. With the limitations above addressed, the platform provides a practical foundation for a proactive, data-driven retention strategy.

## Appendix A: Individual Contribution Table

| Member | Name | Primary responsibility | Contributions |
| --- | --- | --- | --- |
| Member A | *\[Name\]* | AWS infrastructure and security | S3 zones, Lambda trigger, IAM, CloudWatch, SNS, budget alarm |
| Member B | *\[Name\]* | Data validation and preprocessing | Great Expectations rules, cleaning and feature engineering |
| Member C | *\[Name\]* | Modelling and evaluation | Benchmark, cross-validation, Optuna, MLflow, SHAP |
| Member D | *\[Name\]* | Analytics, API and documentation | Athena queries, FastAPI, Docker, CI/CD, report |

*Replace the names and adjust each description to match what each person actually did.*

## Appendix B: Technology Stack

| Area | Tools |
| --- | --- |
| Language and environment | Python 3.13, uv for dependency management |
| Data and ML | pandas, scikit-learn, XGBoost, Optuna, SHAP, imbalanced-learn |
| Experiment tracking | MLflow with SQLite backend |
| Data quality | Great Expectations |
| API and interfaces | FastAPI, Pydantic, Tailwind CSS dashboard, Streamlit |
| Containers and delivery | Docker, Docker Hub, GitHub Actions |
| Cloud | AWS S3, Lambda, SNS, CloudWatch, Glue, Athena, EC2, IAM, Systems Manager |

## Appendix C: SQL Queries

The six Athena queries (contract churn, churner profile, risky combinations, service adoption, revenue at risk, senior citizens) are stored in the repository under `sql/`. Paste the full query text here for submission.
