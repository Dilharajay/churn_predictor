"""Run data validation on the raw dataset."""
import sys
import logging
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from churn_predictor.validation import validate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    raw_path = Path("data/raw/Telco_customer_churn.xlsx")

    if not raw_path.exists():
        logger.error(f"Dataset not found: {raw_path}")
        sys.exit(1)

    logger.info(f"Loading dataset from {raw_path}")
    df = pd.read_excel(raw_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    result = validate_dataset(df)

    if result["success"]:
        logger.info("✅ All validation checks passed!")
        logger.info(f"Dataset is ready for preprocessing ({result['passed_checks']}/{result['total_checks']} checks)")
        # In production: copy to validated/ prefix in S3
    else:
        logger.error("❌ Validation FAILED!")
        logger.error(f"Failed checks ({len(result['failures'])}):")
        for f in result["failures"]:
            logger.error(f"  - {f}")
        # In production: move to rejected/ prefix + SNS alert
        sys.exit(1)


if __name__ == "__main__":
    main()