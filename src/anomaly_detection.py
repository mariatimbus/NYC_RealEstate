#!/usr/bin/env python3
"""
Detectare anomalii — SVD vs K-Means Clusters vs Isolation Forest (AI)
Compară trei metode: reconstruction error SVD, distanță față de centroid K-Means,
și Isolation Forest.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

INPUT_PATH = "data/cleaned_dataset.csv"
RESULTS_DIR = "results"
CHARTS_DIR = "charts"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


def prepare_features(df: pd.DataFrame):
    """Pregătește aceleași feature-uri ca la clustering."""
    feature_cols = [
        "SALE PRICE", "TOTAL UNITS", "RESIDENTIAL UNITS",
        "COMMERCIAL UNITS", "GROSS SQUARE FEET",
        "LAND SQUARE FEET", "YEAR BUILT",
    ]
    X = df[feature_cols].copy()

    for col in feature_cols:
        low, high = X[col].quantile([0.01, 0.99])
        X[col] = X[col].clip(low, high)

    X["SALE_PRICE_LOG"] = np.log1p(X["SALE PRICE"])
    X["GROSS_SQFT_LOG"] = np.log1p(X["GROSS SQUARE FEET"])
    X["LAND_SQFT_LOG"] = np.log1p(X["LAND SQUARE FEET"])

    cluster_features = [
        "SALE_PRICE_LOG", "TOTAL UNITS", "RESIDENTIAL UNITS",
        "COMMERCIAL UNITS", "GROSS_SQFT_LOG", "LAND_SQFT_LOG", "YEAR BUILT",
    ]
    X_cluster = X[cluster_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    return X_scaled, scaler, cluster_features, X_cluster


def svd_anomaly_scores(X_scaled, n_components=5):
    """Calculează reconstruction error folosind SVD."""
    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)

    U_k = U[:, :n_components]
    s_k = s[:n_components]
    Vt_k = Vt[:n_components, :]

    X_reconstructed = U_k @ np.diag(s_k) @ Vt_k
    reconstruction_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

    return reconstruction_error


def kmeans_anomaly_scores(X_scaled, n_clusters=4):
    """K-Means + distanță până la centroid pentru fiecare punct."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_

    # Distanță euclidiană până la centroidul propriului cluster
    distances = np.linalg.norm(X_scaled - centroids[labels], axis=1)

    return distances, labels, kmeans


def isolation_forest_anomalies(X_scaled, contamination=0.05):
    """Rulează Isolation Forest și returnează scoruri + labels."""
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    labels = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    return labels, scores, model


def main():
    print("=" * 60)
    print("DETECTARE ANOMALII — SVD vs K-Means vs Isolation Forest")
    print("=" * 60)

    df = pd.read_csv(INPUT_PATH)
    print(f"Dataset: {len(df):,} rows x {len(df.columns)} cols")

    X_scaled, scaler, feature_names, X_raw = prepare_features(df)
    print(f"Feature matrix shape: {X_scaled.shape}")

    # ── 1. SVD-based anomaly detection ──
    print("\n--- SVD Reconstruction Error ---")
    svd_scores = svd_anomaly_scores(X_scaled, n_components=5)
    svd_threshold = np.percentile(svd_scores, 95)
    svd_anomalies = svd_scores > svd_threshold

    print(f"SVD anomalies detected: {svd_anomalies.sum():,} ({svd_anomalies.mean():.2%})")

    # ── 2. K-Means distance-based anomaly detection ──
    print("\n--- K-Means Cluster Distance ---")
    km_scores, km_labels, km_model = kmeans_anomaly_scores(X_scaled, n_clusters=4)
    km_threshold = np.percentile(km_scores, 95)
    km_anomalies = km_scores > km_threshold

    print(f"K-Means clusters: {np.unique(km_labels)}")
    print(f"K-Means anomalies (top 5% distance): {km_anomalies.sum():,} ({km_anomalies.mean():.2%})")

    # ── 3. Isolation Forest (AI) ──
    print("\n--- Isolation Forest (AI) ---")
    if_labels, if_scores, if_model = isolation_forest_anomalies(X_scaled, contamination=0.05)
    if_anomalies = if_labels == -1

    print(f"Isolation Forest anomalies: {if_anomalies.sum():,} ({if_anomalies.mean():.2%})")

    # ── 4. Comparare între cele 3 metode ──
    print("\n--- Comparare 3 metode ---")

    # Perechi
    svd_if = svd_anomalies & if_anomalies
    svd_km = svd_anomalies & km_anomalies
    km_if = km_anomalies & if_anomalies
    all_three = svd_anomalies & km_anomalies & if_anomalies
    none = ~svd_anomalies & ~km_anomalies & ~if_anomalies

    print(f"Normal (toate 3 agrează):       {none.sum():,} ({none.mean():.2%})")
    print(f"SVD + K-Means (nu IF):          {svd_km.sum() - all_three.sum():,}")
    print(f"SVD + Isolation Forest (nu KM): {svd_if.sum() - all_three.sum():,}")
    print(f"K-Means + Isolation Forest (nu SVD): {km_if.sum() - all_three.sum():,}")
    print(f"Toate 3 metodele (anomalie):    {all_three.sum():,} ({all_three.mean():.2%})")
    print(f"Doar SVD:                       {(svd_anomalies & ~km_anomalies & ~if_anomalies).sum():,}")
    print(f"Doar K-Means:                   {(km_anomalies & ~svd_anomalies & ~if_anomalies).sum():,}")
    print(f"Doar Isolation Forest:          {(if_anomalies & ~svd_anomalies & ~km_anomalies).sum():,}")

    # Overlap percentages
    print(f"\nOverlap SVD ↔ K-Means:  {svd_km.sum() / max(svd_anomalies.sum(), 1) * 100:.1f}%")
    print(f"Overlap SVD ↔ IF:       {svd_if.sum() / max(svd_anomalies.sum(), 1) * 100:.1f}%")
    print(f"Overlap K-Means ↔ IF:   {km_if.sum() / max(km_anomalies.sum(), 1) * 100:.1f}%")

    # Salvează rezultatele
    results_df = pd.DataFrame({
        "SALE_PRICE": df["SALE PRICE"],
        "GROSS_SQUARE_FEET": df["GROSS SQUARE FEET"],
        "NEIGHBORHOOD": df["NEIGHBORHOOD"],
        "CLUSTER": km_labels,
        "SVD_SCORE": svd_scores,
        "SVD_ANOMALY": svd_anomalies,
        "KM_SCORE": km_scores,
        "KM_ANOMALY": km_anomalies,
        "IF_SCORE": if_scores,
        "IF_ANOMALY": if_anomalies,
    })
    results_df.to_csv(os.path.join(RESULTS_DIR, "anomaly_detection_comparison.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/anomaly_detection_comparison.csv")

    # ── 5. Vizualizări ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # (1) Scatter: toate 3 metodele
    ax = axes[0, 0]
    ax.scatter(df["GROSS SQUARE FEET"], df["SALE PRICE"], c="lightgray", s=3, alpha=0.3, label="Normal")

    # Doar SVD
    mask = svd_anomalies & ~km_anomalies & ~if_anomalies
    ax.scatter(df.loc[mask, "GROSS SQUARE FEET"], df.loc[mask, "SALE PRICE"],
               c="blue", s=20, alpha=0.6, label=f"Doar SVD ({mask.sum()})")

    # Doar K-Means
    mask = km_anomalies & ~svd_anomalies & ~if_anomalies
    ax.scatter(df.loc[mask, "GROSS SQUARE FEET"], df.loc[mask, "SALE PRICE"],
               c="orange", s=20, alpha=0.6, label=f"Doar K-Means ({mask.sum()})")

    # Doar IF
    mask = if_anomalies & ~svd_anomalies & ~km_anomalies
    ax.scatter(df.loc[mask, "GROSS SQUARE FEET"], df.loc[mask, "SALE PRICE"],
               c="red", s=20, alpha=0.6, label=f"Doar IF ({mask.sum()})")

    # Ambele / toate 3
    mask = svd_anomalies & km_anomalies & if_anomalies
    ax.scatter(df.loc[mask, "GROSS SQUARE FEET"], df.loc[mask, "SALE PRICE"],
               c="purple", s=40, alpha=0.9, marker="X", label=f"Toate 3 ({mask.sum()})")

    ax.set_xlabel("Gross Square Feet")
    ax.set_ylabel("Sale Price ($)")
    ax.set_title("Anomalii: SVD vs K-Means vs Isolation Forest")
    ax.set_xlim(0, df["GROSS SQUARE FEET"].quantile(0.995))
    ax.set_ylim(0, df["SALE PRICE"].quantile(0.995))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2) Histogramă SVD
    ax = axes[0, 1]
    ax.hist(svd_scores, bins=100, color="steelblue", edgecolor="white", alpha=0.7)
    ax.axvline(svd_threshold, color="red", linestyle="--", label=f"Threshold = {svd_threshold:.1f}")
    ax.set_xlabel("SVD Reconstruction Error")
    ax.set_ylabel("Count")
    ax.set_title("Distribuție SVD Reconstruction Error")
    ax.legend()
    ax.set_xlim(0, np.percentile(svd_scores, 99.5))

    # (3) Histogramă K-Means distances
    ax = axes[0, 2]
    ax.hist(km_scores, bins=100, color="darkorange", edgecolor="white", alpha=0.7)
    ax.axvline(km_threshold, color="red", linestyle="--", label=f"Threshold = {km_threshold:.1f}")
    ax.set_xlabel("Distanță până la centroid (standardizat)")
    ax.set_ylabel("Count")
    ax.set_title("Distribuție K-Means Centroid Distance")
    ax.legend()
    ax.set_xlim(0, np.percentile(km_scores, 99.5))

    # (4) Histogramă Isolation Forest
    ax = axes[1, 0]
    ax.hist(if_scores, bins=100, color="forestgreen", edgecolor="white", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", label="Threshold = 0")
    ax.set_xlabel("Isolation Forest Score (mai mic = mai anomalous)")
    ax.set_ylabel("Count")
    ax.set_title("Distribuție Isolation Forest Scores")
    ax.legend()

    # (5) Bar chart: overlap între metode
    ax = axes[1, 1]
    categories = ["Normal", "Doar\nSVD", "Doar\nK-Means", "Doar\nIF",
                  "SVD+\nKM", "SVD+\nIF", "KM+\nIF", "Toate\n3"]
    counts = [
        none.sum(),
        (svd_anomalies & ~km_anomalies & ~if_anomalies).sum(),
        (km_anomalies & ~svd_anomalies & ~if_anomalies).sum(),
        (if_anomalies & ~svd_anomalies & ~km_anomalies).sum(),
        (svd_anomalies & km_anomalies & ~if_anomalies).sum(),
        (svd_anomalies & if_anomalies & ~km_anomalies).sum(),
        (km_anomalies & if_anomalies & ~svd_anomalies).sum(),
        all_three.sum(),
    ]
    colors = ["lightgray", "blue", "darkorange", "red",
              "purple", "purple", "purple", "black"]
    bars = ax.bar(categories, counts, color=colors, edgecolor="black", alpha=0.7)
    ax.set_ylabel("Număr proprietăți")
    ax.set_title("Overlap detectare anomalii")
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f"{count:,}", ha="center", va="bottom", fontweight="bold", fontsize=8)

    # (6) Scatter K-Means clusters colorat
    ax = axes[1, 2]
    scatter = ax.scatter(df["GROSS SQUARE FEET"], df["SALE PRICE"],
                         c=km_labels, cmap="tab10", s=5, alpha=0.5)
    ax.set_xlabel("Gross Square Feet")
    ax.set_ylabel("Sale Price ($)")
    ax.set_title("K-Means Clusters (k=4)")
    ax.set_xlim(0, df["GROSS SQUARE FEET"].quantile(0.995))
    ax.set_ylim(0, df["SALE PRICE"].quantile(0.995))
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Detectare Anomalii — SVD vs K-Means vs Isolation Forest", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "16_anomaly_detection_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved: {CHARTS_DIR}/16_anomaly_detection_comparison.png")

    # ── 6. Top anomalii confirmate de toate 3 metodele ──
    print("\n--- Top 10 anomalii confirmate (toate 3 metodele) ---")
    top_all = results_df[all_three].nlargest(10, "SVD_SCORE")
    print(top_all[["SALE_PRICE", "GROSS_SQUARE_FEET", "NEIGHBORHOOD", "CLUSTER",
                    "SVD_SCORE", "KM_SCORE", "IF_SCORE"]].to_string(index=False))

    print("\n=== Detectare anomalii finalizată! ===")


if __name__ == "__main__":
    main()
