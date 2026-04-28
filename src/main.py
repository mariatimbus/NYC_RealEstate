#!/usr/bin/env python3
"""NYC Property Sales Dataset — Generate full & simplified CSVs"""

import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

FULL_OUTPUT = "data/full_dataset.csv"
SIMPLIFIED_OUTPUT = "data/simplified_dataset.csv"

# Simplified columns
SIMPLIFIED_COLS = [
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


def load_full_dataset() -> pd.DataFrame:
    """Load the complete dataset from Kaggle."""
    print("Loading full dataset from Kaggle...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "new-york-city/nyc-property-sales",
        "nyc-rolling-sales.csv",
    )
    print(f"Full dataset loaded: {len(df):,} rows × {len(df.columns)} columns")
    return df


def save_full_dataset(df: pd.DataFrame) -> None:
    """Save the raw full dataset."""
    df.to_csv(FULL_OUTPUT, index=False)
    print(f"Full dataset saved to: {FULL_OUTPUT}")


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to numeric, coercing invalid values (like '-') to NaN."""
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")


def build_simplified_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only selected columns and clean numeric fields."""
    sdf = df[SIMPLIFIED_COLS].copy()

    for col in NUMERIC_COLS:
        sdf[col] = clean_numeric(sdf[col])

    # Drop rows with missing values in key fields
    before = len(sdf)
    sdf = sdf.dropna(subset=["SALE PRICE", "GROSS SQUARE FEET", "LAND SQUARE FEET"])
    after = len(sdf)
    print(f"Simplified dataset: kept {after:,} of {before:,} rows after cleaning")

    # Convert integer-like columns
    int_cols = ["TOTAL UNITS", "RESIDENTIAL UNITS", "COMMERCIAL UNITS", "YEAR BUILT"]
    for col in int_cols:
        sdf[col] = sdf[col].astype(int)

    return sdf.reset_index(drop=True)


def save_simplified_dataset(df: pd.DataFrame) -> None:
    """Save the cleaned simplified dataset."""
    df.to_csv(SIMPLIFIED_OUTPUT, index=False)
    print(f"Simplified dataset saved to: {SIMPLIFIED_OUTPUT}")


def remove_old_csvs() -> None:
    """Remove any CSVs other than the two required files."""
    import os
    for f in os.listdir("data"):
        if f.endswith(".csv") and f not in ("full_dataset.csv", "simplified_dataset.csv"):
            os.remove(os.path.join("data", f))
            print(f"Removed old file: data/{f}")


if __name__ == "__main__":
    # 1. Load & save full dataset
    df_full = load_full_dataset()
    save_full_dataset(df_full)

    # 2. Build & save simplified dataset
    df_simple = build_simplified_dataset(df_full)
    save_simplified_dataset(df_simple)

    # 3. Clean up old CSVs
    remove_old_csvs()

    print("\n=== Done ===")
    print(f"Full:       {FULL_OUTPUT}  — {len(df_full):,} rows × {len(df_full.columns)} cols")
    print(f"Simplified: {SIMPLIFIED_OUTPUT} — {len(df_simple):,} rows × {len(df_simple.columns)} cols")
