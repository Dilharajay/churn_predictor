"""
Data validation using Great Expectations.
Implements the quality checks defined in docs/data-card.md §5.
"""
import logging
import great_expectations as gx
import pandas as pd

logger = logging.getLogger(__name__)

# Expected columns (all 33)
EXPECTED_COLUMNS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude", "Gender", "Senior Citizen",
    "Partner", "Dependents", "Tenure Months", "Phone Service",
    "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies", "Contract",
    "Paperless Billing", "Payment Method", "Monthly Charges",
    "Total Charges", "Churn Label", "Churn Value", "Churn Score",
    "CLTV", "Churn Reason",
]

# Value sets from data-card.md §6
VALUE_SETS = {
    "Gender": ["Female", "Male"],
    "Senior Citizen": ["Yes", "No"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "Phone Service": ["Yes", "No"],
    "Multiple Lines": ["No", "No phone service", "Yes"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    "Online Security": ["No", "No internet service", "Yes"],
    "Online Backup": ["No", "No internet service", "Yes"],
    "Device Protection": ["No", "No internet service", "Yes"],
    "Tech Support": ["No", "No internet service", "Yes"],
    "Streaming TV": ["No", "No internet service", "Yes"],
    "Streaming Movies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Paperless Billing": ["Yes", "No"],
    "Payment Method": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
    "Churn Label": ["Yes", "No"],
    "Churn Value": [0, 1],
}


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Run all validation checks against the raw DataFrame.

    Returns:
        dict with keys:
            - 'success': bool — overall pass/fail
            - 'results': list of individual check results
            - 'failures': list of failed check descriptions
    """
    results = []
    failures = []

    # Table-level checks
    results.append(_check("Column count is 33", len(df.columns) == 33))
    results.append(_check(
        "Columns match expected set",
        set(df.columns) == set(EXPECTED_COLUMNS)
    ))
    results.append(_check(
        "Row count between 7000-7100",
        7000 <= len(df) <= 7100
    ))

    # Identity checks
    results.append(_check(
        "CustomerID has no nulls",
        df["CustomerID"].notna().all()
    ))
    results.append(_check(
        "CustomerID is unique",
        df["CustomerID"].is_unique
    ))

    # Categorical value set checks
    for col, valid_values in VALUE_SETS.items():
        actual_values = set(df[col].dropna().unique())
        expected = set(valid_values)
        passed = actual_values.issubset(expected)
        results.append(_check(
            f"{col} values in {valid_values}",
            passed
        ))

    # Numeric range checks
    results.append(_check(
        "Tenure Months in [0, 72]",
        df["Tenure Months"].between(0, 72).all()
    ))
    results.append(_check(
        "Monthly Charges in [0, 200]",
        df["Monthly Charges"].between(0, 200).all()
    ))
    results.append(_check(
        "Churn Score in [0, 100]",
        df["Churn Score"].between(0, 100).all()
    ))

    # Collect failures
    failures = [r["check"] for r in results if not r["passed"]]
    success = len(failures) == 0

    logger.info(f"Validation {'PASSED' if success else 'FAILED'}: "
                f"{sum(r['passed'] for r in results)}/{len(results)} checks passed")

    if failures:
        logger.warning(f"Failed checks: {failures}")

    return {
        "success": success,
        "results": results,
        "failures": failures,
        "total_checks": len(results),
        "passed_checks": sum(r["passed"] for r in results),
    }


def _check(description: str, condition: bool) -> dict:
    """Helper to create a check result."""
    return {"check": description, "passed": bool(condition)}