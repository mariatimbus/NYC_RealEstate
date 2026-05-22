#!/usr/bin/env python3
"""
Detectarea proprietăților subevaluate (undervalued) și supraevaluate (overvalued)
Petrovici Dalia-Teodora

Etapa: calculează reziduurile r = y_real − y_predicted,
       identifică proprietățile cu cele mai mari diferențe față de predicție,
       creează liste cu proprietăți potențial subevaluate și supraevaluate.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

INPUT_PATH = "data/cleaned_dataset.csv"
RESULTS_DIR = "results"
CHARTS_DIR = "charts"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


def load_and_prepare(path: str):
    """
    Încarcă dataset-ul și pregătește feature-urile pentru regresie.
    Folosim One-Hot Encoding pentru variabilele categorice (mai robust decât Label Encoding).
    """
    df = pd.read_csv(path)
    print(f"Dataset încărcat: {len(df):,} rânduri × {len(df.columns)} coloane")

    # Feature-uri numerice
    num_cols = [
        "TOTAL UNITS",
        "RESIDENTIAL UNITS",
        "COMMERCIAL UNITS",
        "GROSS SQUARE FEET",
        "LAND SQUARE FEET",
        "YEAR BUILT",
    ]
    X = df[num_cols].copy()

    # Clip la percentila 1/99 pentru a reduce outlieri extremi
    for col in num_cols:
        low, high = X[col].quantile([0.01, 0.99])
        X[col] = X[col].clip(low, high)

    # Log-transform pentru suprafețe (distribuție skewed)
    X["GROSS_SQFT_LOG"] = np.log1p(X["GROSS SQUARE FEET"])
    X["LAND_SQFT_LOG"] = np.log1p(X["LAND SQUARE FEET"])

    # One-Hot Encoding pentru categorice
    cat_cols = ["NEIGHBORHOOD", "BUILDING CLASS CATEGORY", "BUILDING CLASS AT PRESENT"]
    X_cat = pd.get_dummies(df[cat_cols], columns=cat_cols, drop_first=False)

    # Combinăm numerice + categorice
    X_model = pd.concat([
        X[["TOTAL UNITS", "RESIDENTIAL UNITS", "COMMERCIAL UNITS",
           "GROSS_SQFT_LOG", "LAND_SQFT_LOG", "YEAR BUILT"]],
        X_cat
    ], axis=1)

    # Target: log(SALE PRICE) — stabilizează varianța
    y_actual = df["SALE PRICE"].values
    y_log = np.log1p(y_actual)

    # Standardizare
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_model)

    return df, X_scaled, y_actual, y_log, scaler, list(X_model.columns)


def train_and_predict(X, y_log):
    """
    Antrenează Ridge Regression și returnează predicțiile.
    """
    model = Ridge(alpha=1.0)
    model.fit(X, y_log)
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    return model, y_pred


def compute_residuals(y_actual, y_pred):
    """
    Calculează reziduurile: r = y_real − y_predicted.
    Returnează reziduu absolut ($) și procentual (%).
    """
    residual = y_actual - y_pred
    residual_pct = residual / np.maximum(y_pred, 1) * 100
    return residual, residual_pct


def identify_under_over(df, y_actual, y_pred, residual, residual_pct, n_top=100):
    """
    Identifică top N proprietăți subevaluate și supraevaluate.

    • Undervalued  → reziduu negativ mare (actual << predicted)
    • Overvalued   → reziduu pozitiv mare (actual >> predicted)
    """
    # Sortăm după reziduu
    undervalued_idx = np.argsort(residual)[:n_top]
    overvalued_idx = np.argsort(residual)[-n_top:][::-1]

    # Construim DataFrame-uri
    undervalued = pd.DataFrame({
        "RANK": range(1, n_top + 1),
        "NEIGHBORHOOD": df.iloc[undervalued_idx]["NEIGHBORHOOD"].values,
        "BUILDING_CLASS": df.iloc[undervalued_idx]["BUILDING CLASS CATEGORY"].values,
        "SALE_PRICE": y_actual[undervalued_idx],
        "PREDICTED_PRICE": y_pred[undervalued_idx],
        "RESIDUAL": residual[undervalued_idx],
        "RESIDUAL_PCT": residual_pct[undervalued_idx],
        "GROSS_SQFT": df.iloc[undervalued_idx]["GROSS SQUARE FEET"].values,
        "LAND_SQFT": df.iloc[undervalued_idx]["LAND SQUARE FEET"].values,
        "YEAR_BUILT": df.iloc[undervalued_idx]["YEAR BUILT"].values,
        "TOTAL_UNITS": df.iloc[undervalued_idx]["TOTAL UNITS"].values,
    })

    overvalued = pd.DataFrame({
        "RANK": range(1, n_top + 1),
        "NEIGHBORHOOD": df.iloc[overvalued_idx]["NEIGHBORHOOD"].values,
        "BUILDING_CLASS": df.iloc[overvalued_idx]["BUILDING CLASS CATEGORY"].values,
        "SALE_PRICE": y_actual[overvalued_idx],
        "PREDICTED_PRICE": y_pred[overvalued_idx],
        "RESIDUAL": residual[overvalued_idx],
        "RESIDUAL_PCT": residual_pct[overvalued_idx],
        "GROSS_SQFT": df.iloc[overvalued_idx]["GROSS SQUARE FEET"].values,
        "LAND_SQFT": df.iloc[overvalued_idx]["LAND SQUARE FEET"].values,
        "YEAR_BUILT": df.iloc[overvalued_idx]["YEAR BUILT"].values,
        "TOTAL_UNITS": df.iloc[overvalued_idx]["TOTAL UNITS"].values,
    })

    return undervalued, overvalued


def save_lists(undervalued, overvalued):
    """Salvează listele de subevaluate și supraevaluate în CSV."""
    uv_path = os.path.join(RESULTS_DIR, "undervalued_list.csv")
    ov_path = os.path.join(RESULTS_DIR, "overvalued_list.csv")
    undervalued.to_csv(uv_path, index=False)
    overvalued.to_csv(ov_path, index=False)
    print(f"\n✅ Listă subevaluate salvată: {uv_path}")
    print(f"✅ Listă supraevaluate salvată: {ov_path}")


def print_summary(undervalued, overvalued):
    """Afișează un rezumat al rezultatelor."""
    print("\n" + "=" * 70)
    print("REZUMAT — DETECTARE SUBEVALUATE / SUPRAEVALUATE")
    print("=" * 70)

    print(f"\n📉 TOP 10 PROPRIETĂȚI SUBEVALUATE (Undervalued)")
    print("-" * 70)
    for _, row in undervalued.head(10).iterrows():
        print(f"  #{int(row['RANK'])} {row['NEIGHBORHOOD']:25s} | "
              f"Actual: ${row['SALE_PRICE']:>12,.0f} | "
              f"Predicted: ${row['PREDICTED_PRICE']:>12,.0f} | "
              f"Diff: {row['RESIDUAL_PCT']:>+7.1f}%")

    print(f"\n📈 TOP 10 PROPRIETĂȚI SUPRAEVALUATE (Overvalued)")
    print("-" * 70)
    for _, row in overvalued.head(10).iterrows():
        print(f"  #{int(row['RANK'])} {row['NEIGHBORHOOD']:25s} | "
              f"Actual: ${row['SALE_PRICE']:>12,.0f} | "
              f"Predicted: ${row['PREDICTED_PRICE']:>12,.0f} | "
              f"Diff: {row['RESIDUAL_PCT']:>+7.1f}%")


def plot_residuals(y_actual, y_pred, residual, undervalued, overvalued, df):
    """Generează vizualizări pentru reziduuri și proprietățile identificate."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (1) Actual vs Predicted
    ax = axes[0, 0]
    lim = np.percentile(y_actual, 99)
    m = (y_actual < lim) & (y_pred < lim)
    ax.scatter(y_pred[m], y_actual[m], c="steelblue", s=10, alpha=0.3, edgecolors="none")
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Predicted Price ($)")
    ax.set_ylabel("Actual Price ($)")
    ax.set_title("Actual vs Predicted Sale Price")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) Distribuția reziduurilor
    ax = axes[0, 1]
    ax.hist(residual, bins=150, color="gray", edgecolor="white", alpha=0.7)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(np.percentile(residual, 5), color="green", linestyle="--", lw=1.5, label="Top 5% under")
    ax.axvline(np.percentile(residual, 95), color="red", linestyle="--", lw=1.5, label="Top 5% over")
    ax.set_xlabel("Residual = Actual − Predicted ($)")
    ax.set_ylabel("Count")
    ax.set_title("Distribuția Reziduurilor")
    ax.set_xlim(np.percentile(residual, 1), np.percentile(residual, 99))
    ax.legend()

    # (3) Subevaluate pe scatter
    ax = axes[1, 0]
    ax.scatter(df["GROSS SQUARE FEET"], df["SALE PRICE"], c="lightgray", s=3, alpha=0.3)
    uv_idx = undervalued.index[:50]
    ax.scatter(
        df.iloc[uv_idx]["GROSS SQUARE FEET"],
        df.iloc[uv_idx]["SALE PRICE"],
        c="green", s=25, alpha=0.7, edgecolors="black", linewidth=0.3,
        label="Top 50 Undervalued"
    )
    ax.set_xlabel("Gross Square Feet")
    ax.set_ylabel("Actual Sale Price ($)")
    ax.set_title("Proprietăți Subevaluate")
    ax.set_xlim(0, df["GROSS SQUARE FEET"].quantile(0.995))
    ax.set_ylim(100_000, 200_000_000)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (4) Supraevaluate pe scatter
    ax = axes[1, 1]
    ax.scatter(df["GROSS SQUARE FEET"], df["SALE PRICE"], c="lightgray", s=3, alpha=0.3)
    ov_idx = overvalued.index[:50]
    ax.scatter(
        df.iloc[ov_idx]["GROSS SQUARE FEET"],
        df.iloc[ov_idx]["SALE PRICE"],
        c="red", s=25, alpha=0.7, edgecolors="black", linewidth=0.3,
        label="Top 50 Overvalued"
    )
    ax.set_xlabel("Gross Square Feet")
    ax.set_ylabel("Actual Sale Price ($)")
    ax.set_title("Proprietăți Supraevaluate")
    ax.set_xlim(0, df["GROSS SQUARE FEET"].quantile(0.995))
    ax.set_ylim(100_000, 200_000_000)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Detectare Proprietăți Subevaluate / Supraevaluate — Ridge Regression",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "17_undervalued_overvalued.png"), dpi=150)
    plt.close()
    print(f"\n[OK] Grafic salvat: {CHARTS_DIR}/17_undervalued_overvalued.png")


def save_full_dataset_with_residuals(df, y_pred, residual, residual_pct):
    """Salvează dataset-ul complet cu predicții și reziduuri."""
    output = df.copy()
    output["PREDICTED_PRICE"] = y_pred
    output["RESIDUAL"] = residual
    output["RESIDUAL_PCT"] = residual_pct
    output.to_csv(os.path.join(RESULTS_DIR, "undervalued_overvalued_full.csv"), index=False)
    print(f"✅ Dataset complet salvat: {RESULTS_DIR}/undervalued_overvalued_full.csv")


def main():
    print("=" * 70)
    print("DETECTARE PROPRIETĂȚI SUBEVALUATE / SUPRAEVALUATE")
    print("Petrovici Dalia-Teodora")
    print("=" * 70)

    # 1. Pregătire date
    df, X_scaled, y_actual, y_log, scaler, model_cols = load_and_prepare(INPUT_PATH)

    # Filtrăm outlieri extreme pentru antrenare (transferuri nominale și prețuri aberante)
    # Dar păstrăm predicția pe întreg dataset-ul pentru a prinde și anomaliile
    train_mask = (y_actual >= 100_000) & (y_actual <= 200_000_000)
    print(f"\nFiltrare antrenare: {train_mask.sum():,} / {len(y_actual):,} proprietăți "
          f"($100K – $200M)")

    # 2. Regresie și predicție (antrenare pe filtrat, predicție pe toate)
    print("\nAntrenare Ridge Regression pe log(SALE PRICE)...")
    model = Ridge(alpha=1.0)
    model.fit(X_scaled[train_mask], y_log[train_mask])
    y_pred_log = model.predict(X_scaled)
    y_pred = np.expm1(y_pred_log)
    r2_log = model.score(X_scaled[train_mask], y_log[train_mask])
    print(f"R² (log, pe set filtrat): {r2_log:.4f}")

    # 3. Calculează reziduurile: r = y_real − y_predicted
    print("\nCalcul reziduuri: r = y_real − y_predicted ...")
    residual, residual_pct = compute_residuals(y_actual, y_pred)
    print(f"Reziduu mediu (total): ${residual.mean():,.0f} | Std: ${residual.std():,.0f}")
    print(f"Reziduu mediu (filtrat): ${residual[train_mask].mean():,.0f} | "
          f"Std: ${residual[train_mask].std():,.0f}")

    # 4. Identifică subevaluate și supraevaluate DOAR pe setul filtrat
    print("\nIdentificare top 100 subevaluate și supraevaluate (pe set filtrat)...")
    df_filt = df[train_mask].reset_index(drop=True)
    y_actual_filt = y_actual[train_mask]
    y_pred_filt = y_pred[train_mask]
    residual_filt = residual[train_mask]
    residual_pct_filt = residual_pct[train_mask]

    undervalued, overvalued = identify_under_over(
        df_filt, y_actual_filt, y_pred_filt, residual_filt, residual_pct_filt, n_top=100
    )

    # 5. Afișare rezumat
    print_summary(undervalued, overvalued)

    # 6. Salvare rezultate
    save_lists(undervalued, overvalued)
    save_full_dataset_with_residuals(df, y_pred, residual, residual_pct)

    # 7. Vizualizări (pe setul filtrat)
    print("\nGenerare grafice...")
    plot_residuals(y_actual_filt, y_pred_filt, residual_filt, undervalued, overvalued, df_filt)

    print("\n" + "=" * 70)
    print("ETAPĂ FINALIZATĂ CU SUCCES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
