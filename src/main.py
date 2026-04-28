#!/usr/bin/env python3
"""NYC Property Sales Dataset — Selected Columns Only"""

import pandas as pd

DATASET_PATH = "data/nyc-rolling-sales.csv"

# Columns to keep
COLUMNS = [
    "SALE PRICE",
    "NEIGHBORHOOD",
    "BUILDING CLASS CATEGORY",
    "TOTAL UNITS",
    "RESIDENTIAL UNITS",
    "COMMERCIAL UNITS",
    "GROSS SQUARE FEET",
    "LAND SQUARE FEET",
    "YEAR BUILT",
    "BUILDING CLASS AT PRESENT",
]

# Numeric columns that may contain "-" placeholders
NUMERIC_COLS = [
    "SALE PRICE",
    "GROSS SQUARE FEET",
    "LAND SQUARE FEET",
]


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to numeric, coercing invalid values (like '-') to NaN."""
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")


def load_and_clean_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the dataset, keep only needed columns, and clean numeric fields."""
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path, usecols=COLUMNS)

    for col in NUMERIC_COLS:
        df[col] = clean_numeric(df[col])

    # Drop rows with missing values in key fields
    before = len(df)
    df = df.dropna(subset=["SALE PRICE", "GROSS SQUARE FEET", "LAND SQUARE FEET"])
    after = len(df)
    print(f"Kept {after:,} of {before:,} rows after cleaning")

    # Convert remaining integer-like columns
    int_cols = ["TOTAL UNITS", "RESIDENTIAL UNITS", "COMMERCIAL UNITS", "YEAR BUILT"]
    for col in int_cols:
        df[col] = df[col].astype(int)

    return df.reset_index(drop=True)


def explore_dataset(df: pd.DataFrame) -> None:
    """Print basic exploration of the cleaned dataset."""
    print("\n=== Dataset Info ===")
    print(df.info())
    print("\n=== First 5 Records ===")
    print(df.head())
    print("\n=== Descriptive Statistics ===")
    print(df.describe())


if __name__ == "__main__":
    df = load_and_clean_dataset()
    explore_dataset(df)
