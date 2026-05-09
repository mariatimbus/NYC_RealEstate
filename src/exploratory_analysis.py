#!/usr/bin/env python3
"""
Etapa 2 — Analiză statistică preliminară și clustering (K-Means)
Folosește: data/cleaned_dataset.csv
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

INPUT_PATH = "data/cleaned_dataset.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading dataset from {path}...")
    df = pd.read_csv(path)
    print(f"Dataset loaded: {len(df):,} rows × {len(df.columns)} columns")
    return df


def descriptive_statistics(df: pd.DataFrame) -> None:
    print("\n=== 1. DESCRIPTIVE STATISTICS ===")
    numeric_cols = [
        "SALE PRICE", "TOTAL UNITS", "RESIDENTIAL UNITS",
        "COMMERCIAL UNITS", "GROSS SQUARE FEET",
        "LAND SQUARE FEET", "YEAR BUILT",
    ]
    desc = df[numeric_cols].describe().T
    desc["median"] = df[numeric_cols].median()
    desc["skewness"] = df[numeric_cols].skew()
    print(desc.round(2))
    desc.to_csv(os.path.join(RESULTS_DIR, "descriptive_statistics.csv"))
    print(f"Saved: {RESULTS_DIR}/descriptive_statistics.csv")


def correlation_analysis(df: pd.DataFrame) -> None:
    print("\n=== 2. CORRELATION ANALYSIS ===")
    numeric_cols = [
        "SALE PRICE", "TOTAL UNITS", "RESIDENTIAL UNITS",
        "COMMERCIAL UNITS", "GROSS SQUARE FEET",
        "LAND SQUARE FEET", "YEAR BUILT",
    ]
    corr_matrix = df[numeric_cols].corr()
    print("Correlation with SALE PRICE:")
    print(corr_matrix["SALE PRICE"].drop("SALE PRICE").sort_values(ascending=False).round(4))
    corr_matrix.to_csv(os.path.join(RESULTS_DIR, "correlation_matrix.csv"))
    print(f"Saved: {RESULTS_DIR}/correlation_matrix.csv")


def aggregate_analysis(df: pd.DataFrame) -> None:
    print("\n=== 3. AGGREGATE ANALYSIS ===")

    # By NEIGHBORHOOD
    neighborhood_stats = df.groupby("NEIGHBORHOOD").agg(
        count=("SALE PRICE", "count"),
        avg_price=("SALE PRICE", "mean"),
        median_price=("SALE PRICE", "median"),
        avg_gross_sqft=("GROSS SQUARE FEET", "mean"),
        avg_land_sqft=("LAND SQUARE FEET", "mean"),
        avg_units=("TOTAL UNITS", "mean"),
    ).sort_values("avg_price", ascending=False)
    neighborhood_stats.to_csv(os.path.join(RESULTS_DIR, "neighborhood_stats.csv"))
    print(f"Saved: {RESULTS_DIR}/neighborhood_stats.csv")
    print("Top 5 neighborhoods by average price:")
    print(neighborhood_stats.head().round(2))

    # By BUILDING CLASS CATEGORY
    building_stats = df.groupby("BUILDING CLASS CATEGORY").agg(
        count=("SALE PRICE", "count"),
        avg_price=("SALE PRICE", "mean"),
        median_price=("SALE PRICE", "median"),
        avg_gross_sqft=("GROSS SQUARE FEET", "mean"),
        avg_units=("TOTAL UNITS", "mean"),
    ).sort_values("avg_price", ascending=False)
    building_stats.to_csv(os.path.join(RESULTS_DIR, "building_class_stats.csv"))
    print(f"Saved: {RESULTS_DIR}/building_class_stats.csv")
    print("Top 5 building classes by average price:")
    print(building_stats.head().round(2))


def price_per_sqft_analysis(df: pd.DataFrame) -> None:
    print("\n=== 4. PRICE PER SQUARE FOOT ANALYSIS ===")
    df = df.copy()
    df["PRICE_PER_GROSS_SQFT"] = df["SALE PRICE"] / df["GROSS SQUARE FEET"]
    df["PRICE_PER_LAND_SQFT"] = df["SALE PRICE"] / df["LAND SQUARE FEET"]

    print("Price per GROSS SQFT — mean: {:.2f}, median: {:.2f}".format(
        df["PRICE_PER_GROSS_SQFT"].mean(), df["PRICE_PER_GROSS_SQFT"].median()))
    print("Price per LAND SQFT — mean: {:.2f}, median: {:.2f}".format(
        df["PRICE_PER_LAND_SQFT"].mean(), df["PRICE_PER_LAND_SQFT"].median()))

    # Save distribution stats
    dist_stats = pd.DataFrame({
        "PRICE_PER_GROSS_SQFT": df["PRICE_PER_GROSS_SQFT"].describe(),
        "PRICE_PER_LAND_SQFT": df["PRICE_PER_LAND_SQFT"].describe(),
    })
    dist_stats.to_csv(os.path.join(RESULTS_DIR, "price_per_sqft_stats.csv"))
    print(f"Saved: {RESULTS_DIR}/price_per_sqft_stats.csv")


def kmeans_clustering(df: pd.DataFrame) -> None:
    print("\n=== 5. K-MEANS CLUSTERING ===")

    # Select numeric features for clustering
    feature_cols = [
        "SALE PRICE", "TOTAL UNITS", "RESIDENTIAL UNITS",
        "COMMERCIAL UNITS", "GROSS SQUARE FEET",
        "LAND SQUARE FEET", "YEAR BUILT",
    ]
    X = df[feature_cols].copy()

    # Cap extreme outliers at 1st and 99th percentiles for clustering stability
    for col in feature_cols:
        low, high = X[col].quantile([0.01, 0.99])
        X[col] = X[col].clip(low, high)

    # Log-transform SALE PRICE and sqft to reduce skewness
    X["SALE_PRICE_LOG"] = np.log1p(X["SALE PRICE"])
    X["GROSS_SQFT_LOG"] = np.log1p(X["GROSS SQUARE FEET"])
    X["LAND_SQFT_LOG"] = np.log1p(X["LAND SQUARE FEET"])

    cluster_features = [
        "SALE_PRICE_LOG", "TOTAL UNITS", "RESIDENTIAL UNITS",
        "COMMERCIAL UNITS", "GROSS_SQFT_LOG", "LAND_SQFT_LOG", "YEAR BUILT",
    ]
    X_cluster = X[cluster_features]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Elbow method — compute and print inertia values
    print("Calculating inertia for k=1..10...")
    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
        inertias.append(km.inertia_)

    inertia_df = pd.DataFrame({"k": list(K_range), "inertia": inertias})
    inertia_df.to_csv(os.path.join(RESULTS_DIR, "kmeans_inertia.csv"), index=False)
    print(f"Saved: {RESULTS_DIR}/kmeans_inertia.csv")
    print(inertia_df)

    # Fit K-Means with k=4 (reasonable elbow point)
    optimal_k = 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
    df = df.copy()
    df["CLUSTER"] = cluster_labels

    print(f"\nCluster distribution (k={optimal_k}):")
    print(df["CLUSTER"].value_counts().sort_index())

    # Cluster profiles
    cluster_profile = df.groupby("CLUSTER").agg(
        count=("SALE PRICE", "count"),
        avg_price=("SALE PRICE", "mean"),
        median_price=("SALE PRICE", "median"),
        avg_units=("TOTAL UNITS", "mean"),
        avg_gross_sqft=("GROSS SQUARE FEET", "mean"),
        avg_land_sqft=("LAND SQUARE FEET", "mean"),
        avg_year_built=("YEAR BUILT", "mean"),
    ).round(2)
    print("\nCluster profiles:")
    print(cluster_profile)
    cluster_profile.to_csv(os.path.join(RESULTS_DIR, "cluster_profiles.csv"))
    print(f"Saved: {RESULTS_DIR}/cluster_profiles.csv")

    # PCA for explained variance 
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    print(f"\nPCA explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

    # Save clustered dataset
    df.to_csv(os.path.join(RESULTS_DIR, "clustered_dataset.csv"), index=False)
    print(f"Saved: {RESULTS_DIR}/clustered_dataset.csv")


def main():
    df = load_data(INPUT_PATH)
    descriptive_statistics(df)
    correlation_analysis(df)
    aggregate_analysis(df)
    price_per_sqft_analysis(df)
    kmeans_clustering(df)
    
    print("\n=== Etapa 2 finalizată cu succes! ===")
    print(f"Toate rezultatele sunt salvate în directorul '{RESULTS_DIR}'.")


if __name__ == "__main__":
    main()
