#!/usr/bin/env python3
"""
Identificare proprietăți subevaluate (undervalued) și supraevaluate (overvalued)
folosind Ridge Regression + comparare cu anomaliile SVD / K-Means / Isolation Forest.
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
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


def prepare_features(df: pd.DataFrame):
    """Pregătește feature-uri pentru model."""
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

    cluster_features = [
        "TOTAL UNITS", "RESIDENTIAL UNITS",
        "COMMERCIAL UNITS", "GROSS_SQFT_LOG", "LAND_SQFT_LOG", "YEAR BUILT",
    ]
    X_cluster = X[cluster_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    return X_scaled, scaler, X_cluster


def detect_anomalies(X_scaled):
    """Rulează toți 3 algoritmii de anomalii și returnează flag-urile."""
    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    U_k, s_k, Vt_k = U[:, :5], s[:5], Vt[:5, :]
    X_rec = U_k @ np.diag(s_k) @ Vt_k
    svd_scores = np.sum((X_scaled - X_rec) ** 2, axis=1)
    svd_anomalies = svd_scores > np.percentile(svd_scores, 95)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    centroids = km.cluster_centers_
    km_dist = np.linalg.norm(X_scaled - centroids[km_labels], axis=1)
    km_anomalies = km_dist > np.percentile(km_dist, 95)

    if_model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    if_labels = if_model.fit_predict(X_scaled)
    if_anomalies = if_labels == -1

    return svd_anomalies, km_anomalies, if_anomalies, km_labels


def analyze_and_print(df, X_scaled, y_actual, y_log, subset_name, min_price=0, max_price=float('inf')):
    """Rulează regresie Ridge, identifică under/over-valued și printează rezultate."""
    mask = (y_actual >= min_price) & (y_actual <= max_price)
    if mask.sum() < 100:
        print(f"[{subset_name}] Prea puține date ({mask.sum()}), sărim.")
        return None, None, None

    X_sub = X_scaled[mask]
    y_log_sub = y_log[mask]
    y_actual_sub = y_actual[mask]
    df_sub = df[mask].reset_index(drop=True)

    model = Ridge(alpha=1.0)
    model.fit(X_sub, y_log_sub)
    y_pred_log = model.predict(X_sub)
    y_pred = np.expm1(y_pred_log)

    residual = y_actual_sub - y_pred
    residual_pct = residual / np.maximum(y_pred, 1) * 100

    print(f"\n=== {subset_name} ===")
    print(f"Samples: {mask.sum():,} | R² (log): {model.score(X_sub, y_log_sub):.4f}")
    print(f"Mean residual: ${residual.mean():,.0f} | Std: ${residual.std():,.0f}")

    n_top = 30
    undervalued_idx = np.argsort(residual)[:n_top]
    overvalued_idx = np.argsort(residual)[-n_top:][::-1]

    print(f"\n--- Top {n_top} UNDERVALUED ({subset_name}) ---")
    for i in undervalued_idx[:10]:
        print(f"  {df_sub.iloc[i]['NEIGHBORHOOD']:25s} | "
              f"Actual: ${y_actual_sub[i]:>12,.0f} | "
              f"Predicted: ${y_pred[i]:>12,.0f} | "
              f"Diff: {residual_pct[i]:>7.1f}% | "
              f"{df_sub.iloc[i]['BUILDING CLASS CATEGORY'][:30]}")

    print(f"\n--- Top {n_top} OVERVALUED ({subset_name}) ---")
    for i in overvalued_idx[:10]:
        print(f"  {df_sub.iloc[i]['NEIGHBORHOOD']:25s} | "
              f"Actual: ${y_actual_sub[i]:>12,.0f} | "
              f"Predicted: ${y_pred[i]:>12,.0f} | "
              f"Diff: {residual_pct[i]:>+7.1f}% | "
              f"{df_sub.iloc[i]['BUILDING CLASS CATEGORY'][:30]}")

    return residual, residual_pct, df_sub, y_pred


def main():
    print("=" * 60)
    print("UNDERVALUED / OVERVALUED — Ridge + Anomalies")
    print("=" * 60)

    df = pd.read_csv(INPUT_PATH)
    print(f"Dataset: {len(df):,} rows")

    X_scaled, scaler, X_raw = prepare_features(df)
    y_actual = df["SALE PRICE"].values
    y_log = np.log1p(y_actual)

    # Anomalii pe întreg dataset-ul
    svd_a, km_a, if_a, km_labels = detect_anomalies(X_scaled)

    # ── Analiză 1: Toate datele (raw) ──
    res_all, pct_all, df_all, pred_all = analyze_and_print(
        df, X_scaled, y_actual, y_log, "TOATE DATELE"
    )

    # ── Analiză 2: Fără transferuri nominale și extreme ──
    # Eliminăm prețurile < 100K (transferuri nominale) și > 200M (extreme outlieri)
    res_filt, pct_filt, df_filt, pred_filt = analyze_and_print(
        df, X_scaled, y_actual, y_log, "FILTRAT ($100K – $200M)",
        min_price=100_000, max_price=200_000_000
    )

    # ── Analiză 3: Per cluster K-Means (preț relativ la cluster) ──
    print(f"\n{'='*60}")
    print("ANALIZĂ PE CLUSTER K-Means (preț relativ la mediana clusterului)")
    print(f"{'='*60}")

    df_cluster = df.copy()
    df_cluster["CLUSTER"] = km_labels
    df_cluster["PRICE_PER_SQFT"] = df_cluster["SALE PRICE"] / df_cluster["GROSS SQUARE FEET"].clip(lower=1)

    cluster_medians = df_cluster.groupby("CLUSTER")["PRICE_PER_SQFT"].median()
    print("Mediana $/sqft per cluster:")
    for c, m in cluster_medians.items():
        print(f"  Cluster {c}: ${m:,.2f}/sqft")

    df_cluster["RELATIVE_TO_CLUSTER"] = (
        df_cluster["PRICE_PER_SQFT"] - df_cluster["CLUSTER"].map(cluster_medians)
    ) / df_cluster["CLUSTER"].map(cluster_medians) * 100

    print(f"\n--- Top 10 UNDERVALUED per cluster (preț/sqft sub mediana clusterului) ---")
    uv_cluster = df_cluster.nsmallest(10, "RELATIVE_TO_CLUSTER")
    for _, row in uv_cluster.iterrows():
        print(f"  Cluster {int(row['CLUSTER'])} | {row['NEIGHBORHOOD']:25s} | "
              f"${row['SALE PRICE']:>10,.0f} | {row['GROSS SQUARE FEET']:>8,.0f} sqft | "
              f"${row['PRICE_PER_SQFT']:>7.2f}/sqft | {row['RELATIVE_TO_CLUSTER']:>+6.1f}% vs mediana cluster")

    print(f"\n--- Top 10 OVERVALUED per cluster (preț/sqft peste mediana clusterului) ---")
    ov_cluster = df_cluster.nlargest(10, "RELATIVE_TO_CLUSTER")
    for _, row in ov_cluster.iterrows():
        print(f"  Cluster {int(row['CLUSTER'])} | {row['NEIGHBORHOOD']:25s} | "
              f"${row['SALE PRICE']:>10,.0f} | {row['GROSS SQUARE FEET']:>8,.0f} sqft | "
              f"${row['PRICE_PER_SQFT']:>7.2f}/sqft | {row['RELATIVE_TO_CLUSTER']:>+6.1f}% vs mediana cluster")

    # ── Salvare CSV combinat ──
    output = pd.DataFrame({
        "NEIGHBORHOOD": df["NEIGHBORHOOD"],
        "BUILDING_CLASS": df["BUILDING CLASS CATEGORY"],
        "SALE_PRICE": y_actual,
        "GROSS_SQFT": df["GROSS SQUARE FEET"],
        "CLUSTER": km_labels,
        "PRICE_PER_SQFT": y_actual / df["GROSS SQUARE FEET"].clip(lower=1),
        "SVD_ANOMALY": svd_a,
        "KM_ANOMALY": km_a,
        "IF_ANOMALY": if_a,
    })
    output.to_csv(os.path.join(RESULTS_DIR, "undervalued_overvalued.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/undervalued_overvalued.csv")

    # ── Vizualizări ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (1) Actual vs Predicted (filtrat)
    if res_filt is not None:
        ax = axes[0, 0]
        y_act = df_filt["SALE PRICE"].values
        lim = np.percentile(y_act, 99)
        m = (y_act < lim) & (pred_filt < lim)
        ax.scatter(pred_filt[m], y_act[m], c="steelblue", s=10, alpha=0.3, edgecolors="none")
        ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect")
        ax.set_xlabel("Predicted Price ($)")
        ax.set_ylabel("Actual Price ($)")
        ax.set_title("Actual vs Predicted (Filtrat $100K–$200M)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # (2) Residual distribution (filtrat)
    if res_filt is not None:
        ax = axes[0, 1]
        ax.hist(res_filt, bins=150, color="gray", edgecolor="white", alpha=0.7)
        ax.axvline(0, color="black", lw=1)
        ax.axvline(np.percentile(res_filt, 5), color="green", linestyle="--", lw=1.5, label="Top 5% under")
        ax.axvline(np.percentile(res_filt, 95), color="red", linestyle="--", lw=1.5, label="Top 5% over")
        ax.set_xlabel("Residual = Actual − Predicted ($)")
        ax.set_ylabel("Count")
        ax.set_title("Distribuție Residuals (Filtrat)")
        ax.set_xlim(np.percentile(res_filt, 2), np.percentile(res_filt, 98))
        ax.legend()

    # (3) Undervalued pe scatter
    ax = axes[1, 0]
    ax.scatter(df["GROSS SQUARE FEET"], df["SALE PRICE"], c="lightgray", s=3, alpha=0.3)
    # Top 50 subevaluate din setul filtrat
    if res_filt is not None:
        uv_idx = np.argsort(res_filt)[:50]
        ax.scatter(df_filt.iloc[uv_idx]["GROSS SQUARE FEET"], df_filt.iloc[uv_idx]["SALE PRICE"],
                   c="green", s=25, alpha=0.7, edgecolors="black", linewidth=0.3, label="Top 50 Undervalued")
    ax.set_xlabel("Gross Square Feet")
    ax.set_ylabel("Actual Sale Price ($)")
    ax.set_title("Undervalued Properties (Filtrat)")
    ax.set_xlim(0, df["GROSS SQUARE FEET"].quantile(0.995))
    ax.set_ylim(100_000, 200_000_000)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (4) Overvalued pe scatter
    ax = axes[1, 1]
    ax.scatter(df["GROSS SQUARE FEET"], df["SALE PRICE"], c="lightgray", s=3, alpha=0.3)
    if res_filt is not None:
        ov_idx = np.argsort(res_filt)[-50:][::-1]
        ax.scatter(df_filt.iloc[ov_idx]["GROSS SQUARE FEET"], df_filt.iloc[ov_idx]["SALE PRICE"],
                   c="red", s=25, alpha=0.7, edgecolors="black", linewidth=0.3, label="Top 50 Overvalued")
    ax.set_xlabel("Gross Square Feet")
    ax.set_ylabel("Actual Sale Price ($)")
    ax.set_title("Overvalued Properties (Filtrat)")
    ax.set_xlim(0, df["GROSS SQUARE FEET"].quantile(0.995))
    ax.set_ylim(100_000, 200_000_000)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Underpriced vs Overpriced — Ridge Regression + Clusters", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "17_undervalued_overvalued.png"), dpi=150)
    plt.close()
    print(f"Saved: {CHARTS_DIR}/17_undervalued_overvalued.png")

    print("\n=== Analiză finalizată! ===")


if __name__ == "__main__":
    main()
