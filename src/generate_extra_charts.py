#!/usr/bin/env python3
"""
Grafice extra — leagă anomaly detection, clusters, SVD și under/over-valued.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

INPUT_PATH = "data/cleaned_dataset.csv"
RESULTS_DIR = "results"
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)


def prepare(df):
    feature_cols = [
        "SALE PRICE", "TOTAL UNITS", "RESIDENTIAL UNITS",
        "COMMERCIAL UNITS", "GROSS SQUARE FEET",
        "LAND SQUARE FEET", "YEAR BUILT",
    ]
    X = df[feature_cols].copy()
    for col in feature_cols:
        low, high = X[col].quantile([0.01, 0.99])
        X[col] = X[col].clip(low, high)
    X["GROSS_SQFT_LOG"] = np.log1p(X["GROSS SQUARE FEET"])
    X["LAND_SQFT_LOG"] = np.log1p(X["LAND SQUARE FEET"])

    features = [
        "TOTAL UNITS", "RESIDENTIAL UNITS",
        "COMMERCIAL UNITS", "GROSS_SQFT_LOG", "LAND_SQFT_LOG", "YEAR BUILT",
    ]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X[features])
    return Xs, scaler, features, X


def main():
    df = pd.read_csv(INPUT_PATH)
    X_scaled, scaler, feature_names, X_raw = prepare(df)
    y_actual = df["SALE PRICE"].values
    y_log = np.log1p(y_actual)

    # ── Modele ──
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y_log)
    y_pred = np.expm1(ridge.predict(X_scaled))

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)

    if_model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    if_labels = if_model.fit_predict(X_scaled)

    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    X_rec = U[:, :5] @ np.diag(s[:5]) @ Vt[:5, :]
    svd_err = np.sum((X_scaled - X_rec) ** 2, axis=1)
    svd_anom = svd_err > np.percentile(svd_err, 95)

    df_plot = pd.DataFrame({
        "price": y_actual,
        "pred": y_pred,
        "sqft": df["GROSS SQUARE FEET"],
        "year": df["YEAR BUILT"],
        "neighborhood": df["NEIGHBORHOOD"],
        "building_class": df["BUILDING CLASS CATEGORY"],
        "cluster": km_labels,
        "if_anomaly": if_labels == -1,
        "svd_anomaly": svd_anom,
    })
    df_plot["residual"] = df_plot["price"] - df_plot["pred"]
    df_plot["residual_pct"] = df_plot["residual"] / np.maximum(df_plot["pred"], 1) * 100

    # ── Grafic 1: Feature importance Ridge ──
    fig, ax = plt.subplots(figsize=(10, 5))
    coefs = ridge.coef_
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    colors = ["green" if c > 0 else "red" for c in coefs[sorted_idx]]
    ax.barh(np.array(feature_names)[sorted_idx], coefs[sorted_idx], color=colors, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Coeficient Ridge (impact asupra log(preț))")
    ax.set_title("Feature Importance — Ridge Regression")
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "18_ridge_feature_importance.png"), dpi=150)
    plt.close()
    print("Saved: charts/18_ridge_feature_importance.png")

    # ── Grafic 2: Anomaly rate per cluster ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    methods = [("svd_anomaly", "SVD"), ("if_anomaly", "Isolation Forest"), ("svd_anomaly", "Both")]

    for ax, (col, title) in zip(axes, [("svd_anomaly", "SVD"), ("if_anomaly", "Isolation Forest"),
                                        ("svd_anomaly", "SVD + IF overlap")]):
        if title == "SVD + IF overlap":
            rates = df_plot.groupby("cluster").apply(lambda g: (g["svd_anomaly"] & g["if_anomaly"]).mean() * 100)
        else:
            rates = df_plot.groupby("cluster")[col].mean() * 100
        bars = ax.bar(rates.index, rates.values, color=["steelblue", "darkorange", "forestgreen", "crimson"],
                      edgecolor="black", alpha=0.8)
        ax.set_xlabel("Cluster K-Means")
        ax.set_ylabel("Anomaly Rate (%)")
        ax.set_title(f"{title}")
        ax.set_ylim(0, max(rates.values) * 1.2)
        for bar, v in zip(bars, rates.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.suptitle("Anomaly Rate per K-Means Cluster", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "19_anomaly_rate_by_cluster.png"), dpi=150)
    plt.close()
    print("Saved: charts/19_anomaly_rate_by_cluster.png")

    # ── Grafic 3: Under/Over/Anomaly overlay ──
    fig, ax = plt.subplots(figsize=(12, 7))

    # Normal
    ax.scatter(df_plot["sqft"], df_plot["price"], c="lightgray", s=4, alpha=0.3, label="Normal")

    # Anomalii (toate 3 metode)
    mask_all = df_plot["svd_anomaly"] & df_plot["if_anomaly"]
    ax.scatter(df_plot.loc[mask_all, "sqft"], df_plot.loc[mask_all, "price"],
               c="purple", s=30, alpha=0.7, marker="X", label=f"Anomalii confirmate ({mask_all.sum()})")

    # Undervalued (filtrat $100K-$200M, top 5% residual negativ)
    mask_uv = (df_plot["price"] >= 100_000) & (df_plot["price"] <= 200_000_000) & \
              (df_plot["residual"] < np.percentile(df_plot["residual"], 5))
    ax.scatter(df_plot.loc[mask_uv, "sqft"], df_plot.loc[mask_uv, "price"],
               c="green", s=20, alpha=0.6, edgecolors="darkgreen", linewidth=0.5,
               label=f"Undervalued ({mask_uv.sum()})")

    # Overvalued
    mask_ov = (df_plot["price"] >= 100_000) & (df_plot["price"] <= 200_000_000) & \
              (df_plot["residual"] > np.percentile(df_plot["residual"], 95))
    ax.scatter(df_plot.loc[mask_ov, "sqft"], df_plot.loc[mask_ov, "price"],
               c="red", s=20, alpha=0.6, edgecolors="darkred", linewidth=0.5,
               label=f"Overvalued ({mask_ov.sum()})")

    ax.set_xlabel("Gross Square Feet")
    ax.set_ylabel("Sale Price ($)")
    ax.set_title("Underpriced vs Overpriced vs Anomalies")
    ax.set_xlim(0, df_plot["sqft"].quantile(0.995))
    ax.set_ylim(100_000, 200_000_000)
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "20_under_over_anomaly_overlay.png"), dpi=150)
    plt.close()
    print("Saved: charts/20_under_over_anomaly_overlay.png")

    # ── Grafic 4: SVD Reconstruction Error vs Actual Price ──
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_plot["price"], svd_err, c=df_plot["cluster"], cmap="tab10",
                         s=8, alpha=0.5, edgecolors="none")
    ax.axhline(np.percentile(svd_err, 95), color="red", linestyle="--", lw=1.5,
               label=f"Anomaly threshold (95th pct)")
    ax.set_xlabel("Actual Sale Price ($)")
    ax.set_ylabel("SVD Reconstruction Error")
    ax.set_title("SVD Reconstruction Error vs Sale Price")
    ax.set_xlim(0, np.percentile(df_plot["price"], 99))
    ax.set_ylim(0, np.percentile(svd_err, 99.5))
    ax.legend()
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "21_svd_error_vs_price.png"), dpi=150)
    plt.close()
    print("Saved: charts/21_svd_error_vs_price.png")

    # ── Grafic 5: Price vs Year Built cu cluster colors + anomaly ring ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for cluster_id in sorted(df_plot["cluster"].unique()):
        sub = df_plot[df_plot["cluster"] == cluster_id]
        ax.scatter(sub["year"], sub["price"], s=8, alpha=0.4, label=f"Cluster {cluster_id}")
    # Anomalii ca cercuri goale
    anom = df_plot[df_plot["svd_anomaly"] & df_plot["if_anomaly"]]
    ax.scatter(anom["year"], anom["price"], facecolors="none", edgecolors="black",
               s=60, linewidth=1.2, label=f"Anomalii ({len(anom)})")
    ax.set_xlabel("Year Built")
    ax.set_ylabel("Sale Price ($)")
    ax.set_title("Price vs Year Built — Clusters & Anomalies")
    ax.set_ylim(1, df_plot["price"].quantile(0.995))
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "22_price_vs_year_clusters.png"), dpi=150)
    plt.close()
    print("Saved: charts/22_price_vs_year_clusters.png")

    # ── Grafic 6: Boxplot preț pe cluster ──
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_data = [df_plot[df_plot["cluster"] == c]["price"].values for c in sorted(df_plot["cluster"].unique())]
    bp = ax.boxplot(cluster_data, labels=[f"Cluster {c}" for c in sorted(df_plot["cluster"].unique())],
                    patch_artist=True, showfliers=False)
    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Sale Price ($)")
    ax.set_title("Price Distribution per K-Means Cluster")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "23_price_boxplot_by_cluster.png"), dpi=150)
    plt.close()
    print("Saved: charts/23_price_boxplot_by_cluster.png")

    print("\n=== Toate graficele extra generate! ===")


if __name__ == "__main__":
    main()
