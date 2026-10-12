"""Data preprocessing pipeline — mirrors notebook logic in a reusable module."""
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from . import config

logger = logging.getLogger(__name__)


def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw Excel dataset."""
    if path is None:
        path = config.RAW_DIR / config.RAW_FILENAME
    logger.info(f"Loading raw data from {path}")
    df = pd.read_excel(path)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns per data-card.md."""
    logger.info(f"Dropping {len(config.DROP_COLUMNS)} columns")
    return df.drop(columns=config.DROP_COLUMNS, errors="ignore")


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce Total Charges from string to numeric, fill NaN with median."""
    df = df.copy()
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    n_null = df["Total Charges"].isna().sum()
    if n_null > 0:
        median_val = df["Total Charges"].median()
        df["Total Charges"] = df["Total Charges"].fillna(median_val)
        logger.info(f"Filled {n_null} NaN values in Total Charges with median ({median_val:.2f})")

    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode binary columns, one-hot-encode multi-class columns."""
    df = df.copy()

    # Label encoding for binary columns
    le = LabelEncoder()
    for col in config.BINARY_CATEGORICAL:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            logger.debug(f"Label encoded: {col}")

    # One-hot encoding for multi-class columns
    multi_cols = [c for c in config.MULTI_CATEGORICAL if c in df.columns]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True, dtype=int)
    logger.info(f"One-hot encoded {len(multi_cols)} columns → {len(df.columns)} total columns")

    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply StandardScaler and MinMaxScaler to numeric features."""
    df = df.copy()

    std_cols = [c for c in config.STANDARD_SCALED if c in df.columns]
    if std_cols:
        scaler = StandardScaler()
        df[std_cols] = scaler.fit_transform(df[std_cols])
        logger.info(f"StandardScaled: {std_cols}")

    mm_cols = [c for c in config.MINMAX_SCALED if c in df.columns]
    if mm_cols:
        scaler = MinMaxScaler()
        df[mm_cols] = scaler.fit_transform(df[mm_cols])
        logger.info(f"MinMaxScaled: {mm_cols}")

    return df


def split_data(df: pd.DataFrame):
    """Split into train/test sets."""
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    logger.info(f"Split: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test


def save_processed(X_train, X_test, y_train, y_test, output_dir=None):
    """Save processed datasets as CSVs."""
    if output_dir is None:
        output_dir = config.PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    logger.info(f"Saved processed data to {output_dir}")


def run_pipeline(raw_path=None):
    """Execute the full preprocessing pipeline."""
    df = load_raw_data(raw_path)
    df = drop_columns(df)
    df = fix_data_types(df)
    df = encode_categorical(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    save_processed(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test