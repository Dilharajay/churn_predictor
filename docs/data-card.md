---
title: "Dataset Card & Data-Quality Plan (Great Expectations)"
dataset: yeanzc/telco-customer-churn-ibm-dataset
rows: 7043
columns: 33
file: data/raw/Telco_customer_churn.xlsx
format: xlsx
---

# Dataset Card & Data-Quality Plan

The authoritative reference for the churn dataset and how we validate it with
**Great Expectations**. Values below are verified against the actual file, not the
(slightly different) IBM blog description.

## 1. Overview

| Property | Value |
|----------|-------|
| Source | kagglehub `yeanzc/telco-customer-churn-ibm-dataset` |
| Description | Fictional California telco; home phone + Internet, Q3 |
| Rows | 7,043 |
| Columns | 33 |
| Local file | `data/raw/Telco_customer_churn.xlsx` |
| Target | `Churn Value` (1 = churned, 0 = retained) |

## 2. Schema (verified)

| # | Column | dtype | Role |
|---|--------|-------|------|
| 1 | CustomerID | str | ID |
| 2 | Count | int64 | drop (reporting artifact) |
| 3 | Country | str | drop (geography) |
| 4 | State | str | drop (geography) |
| 5 | City | str | drop (geography, high cardinality) |
| 6 | Zip Code | int64 | drop (geography) |
| 7 | Lat Long | str | drop (redundant) |
| 8 | Latitude | float64 | drop (geography) |
| 9 | Longitude | float64 | drop (geography) |
| 10 | Gender | str | categorical feature |
| 11 | Senior Citizen | str | categorical feature |
| 12 | Partner | str | categorical feature |
| 13 | Dependents | str | categorical feature |
| 14 | Tenure Months | int64 | numeric feature |
| 15 | Phone Service | str | categorical feature |
| 16 | Multiple Lines | str | categorical feature |
| 17 | Internet Service | str | categorical feature |
| 18 | Online Security | str | categorical feature |
| 19 | Online Backup | str | categorical feature |
| 20 | Device Protection | str | categorical feature |
| 21 | Tech Support | str | categorical feature |
| 22 | Streaming TV | str | categorical feature |
| 23 | Streaming Movies | str | categorical feature |
| 24 | Contract | str | categorical feature |
| 25 | Paperless Billing | str | categorical feature |
| 26 | Payment Method | str | categorical feature |
| 27 | Monthly Charges | float64 | numeric feature |
| 28 | Total Charges | object (string) | numeric feature (coerced) |
| 29 | Churn Label | str | drop (redundant with target) |
| 30 | Churn Value | int64 | **target** (0/1) |
| 31 | Churn Score | int64 | drop (leakage) |
| 32 | CLTV | int64 | numeric feature |
| 33 | Churn Reason | str | drop (leakage — only known for churners) |

## 3. Feature engineering

- **Target:** `Churn Value` (0/1).
- **Numeric features:** `Tenure Months`, `Monthly Charges`, `Total Charges`, `CLTV`.
- **Categorical features (16):** Gender, Senior Citizen, Partner, Dependents, Phone Service,
  Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection,
  Tech Support, Streaming TV, Streaming Movies, Contract, Paperless Billing, Payment Method.
- **Dropped:** Count, Country, State, City, Zip Code, Lat Long, Latitude, Longitude,
  Churn Label, Churn Score, Churn Reason.
- **Note:** `Total Charges` is string-typed in the raw file (empty strings for tenure-0
  rows); it must be coerced with `pd.to_numeric(errors="coerce")` in preprocessing.

## 4. Data card vs. actual file (discrepancies)

The IBM blog data card differs from the actual file. Use the file values:

| Data card says | Actual file |
|----------------|-------------|
| Monthly Charge | **Monthly Charges** (plural) |
| Internet Service: No, DSL, Fiber Optic, Cable | **DSL, Fiber optic, No** (no "Cable") |
| Contract: Month-to-Month, One Year, Two Year | **Month-to-month, One year, Two year** |
| Payment Method: Bank Withdrawal, Credit Card, Mailed Check | **Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check** |
| Multiple Lines / Online* / Streaming* / Tech Support: Yes, No | also include **No phone service** / **No internet service** |
| Total Charges numeric | **string** (empty strings for tenure-0 rows) |

## 5. Great Expectations data-quality plan

A GX 1.x expectation suite gates the pipeline (as the Dagster `validate` asset):

**Table level**
- `ExpectTableColumnsToMatchSet` — exactly the 33 columns.
- `ExpectTableRowCountToBeBetween` — 7,000–7,100 rows.

**Identity**
- `ExpectColumnValuesToNotBeNull` and `ExpectColumnValuesToBeUnique` on `CustomerID`.

**Targets**
- `ExpectColumnValuesToBeInSet` on `Churn Label` (Yes/No) and `Churn Value` (0/1).

**Categorical enums** (one `ExpectColumnValuesToBeInSet` per column, using §6 value sets).

**Numeric ranges**
- Tenure Months 0–72; Monthly Charges 0–200; Churn Score 0–100; Latitude −90…90; Longitude −180…180.
- `Total Charges` is not range-checked (string-typed); it is coerced in preprocessing.

**Integration:** the suite runs inside the Dagster pipeline; a failure stops
`validated_dataset` and routes the issue to `quarantine/` + an SNS alert.

## 6. Verified value sets

| Column | Values |
|--------|--------|
| Gender | Female, Male |
| Senior Citizen | Yes, No |
| Partner | Yes, No |
| Dependents | Yes, No |
| Phone Service | Yes, No |
| Multiple Lines | No, No phone service, Yes |
| Internet Service | DSL, Fiber optic, No |
| Online Security | No, No internet service, Yes |
| Online Backup | No, No internet service, Yes |
| Device Protection | No, No internet service, Yes |
| Tech Support | No, No internet service, Yes |
| Streaming TV | No, No internet service, Yes |
| Streaming Movies | No, No internet service, Yes |
| Contract | Month-to-month, One year, Two year |
| Paperless Billing | Yes, No |
| Payment Method | Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check |
| Churn Label | Yes, No |
| Churn Value | 0, 1 |
