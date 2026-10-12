# Run this once to create the Athena-ready CSV
import pandas as pd
from churn_predictor.preprocessing import load_raw_data, drop_columns, fix_data_types
df = load_raw_data()
df = fix_data_types(df)
# Keep readable columns (don't encode) for SQL analysis
cols_to_keep = [
    "Gender", "Senior Citizen", "Partner", "Dependents",
    "Tenure Months", "Phone Service", "Multiple Lines",
    "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV",
    "Streaming Movies", "Contract", "Paperless Billing",
    "Payment Method", "Monthly Charges", "Total Charges",
    "CLTV", "Churn Value"
]
df_athena = df[cols_to_keep]
df_athena.to_csv("data/processed/churn_data_athena.csv", index=False)