#!/usr/bin/env python3
"""NYC Property Sales Dataset Loader"""

import pandas as pd

DATASET_PATH = "data/nyc-rolling-sales.csv"


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the NYC Property Sales dataset from the local CSV."""
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """Print basic exploration of the dataset."""
    print("\n=== Dataset Info ===")
    print(df.info())
    print("\n=== First 5 Records ===")
    print(df.head())
    print("\n=== Column Names ===")
    print(list(df.columns))


if __name__ == "__main__":
    df = load_dataset()
    explore_dataset(df)
