#!/usr/bin/env python3
"""
SVD manual pentru analiza proprietăților imobiliare NYC.

Interpretare conceptuală:
  A = U · Σ · Vᵀ

  • U   — preferințele pieței / cumpărătorilor
          (SALE PRICE, NEIGHBORHOOD, BUILDING CLASS CATEGORY)
  • Σ   — importanța componentelor  
          (TOTAL UNITS, RESIDENTIAL UNITS, COMMERCIAL UNITS)
  • Vᵀ  — relațiile dintre caracteristicile proprietăților
          (GROSS SQUARE FEET, LAND SQUARE FEET, YEAR BUILT, BUILDING CLASS AT PRESENT)

Implementare manuală folosind descompunerea în valori proprii pe AᵀA.
"""

import os
import warnings
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

INPUT_PATH = "data/cleaned_dataset.csv"
RESULTS_DIR = "results"
CHARTS_DIR = "charts"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helperi manuali (fără np.funcții de matematică)
# ---------------------------------------------------------------------------

def _norm(x):
    """Norma Euclidiană (L2) manuală."""
    return math.sqrt(sum(float(xi) ** 2 for xi in x))


def _argsort_desc(arr):
    """Argsort descrescător manual."""
    indexed = [(float(arr[i]), i) for i in range(len(arr))]
    indexed.sort(key=lambda t: t[0], reverse=True)
    return [idx for _, idx in indexed]


def _diag_extract(A):
    """Extrage diagonala principală ca listă."""
    n = min(A.shape[0], A.shape[1])
    return [float(A[i, i]) for i in range(n)]


def _diag_matrix(d):
    """Creează matrice diagonală din listă."""
    n = len(d)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = d[i]
    return A


def _cumsum(arr):
    """Sumă cumulativă manuală."""
    result = []
    s = 0.0
    for x in arr:
        s += x
        result.append(s)
    return result


def _randn(n):
    """Generează n valori N(0,1) folosind Box-Muller."""
    result = []
    while len(result) < n:
        u1 = random.random()
        u2 = random.random()
        if u1 == 0:
            continue
        mag = math.sqrt(-2.0 * math.log(u1))
        z1 = mag * math.cos(2.0 * math.pi * u2)
        z2 = mag * math.sin(2.0 * math.pi * u2)
        result.append(z1)
        if len(result) < n:
            result.append(z2)
    return np.array(result)


def _outer(u, v):
    """Produs exterior manual."""
    m, n = len(u), len(v)
    A = np.zeros((m, n))
    for i in range(m):
        ui = float(u[i])
        for j in range(n):
            A[i, j] = ui * float(v[j])
    return A


# 1. Încărcare & preprocesare


def load_and_preprocess(path: str):
    """Încarcă datele și encodează variabilele categorice."""
    df = pd.read_csv(path)
    print(f"Dataset încărcat: {len(df):,} rânduri × {len(df.columns)} coloane")

    # Encodează categoricele cu Label Encoding (păstrăm matricea numerică & compactă)
    cat_cols = ["NEIGHBORHOOD", "BUILDING CLASS CATEGORY", "BUILDING CLASS AT PRESENT"]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes

    # Selectăm doar coloanele de interes, în ordinea grupurilor
    feature_cols = [
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
    df = df[feature_cols].dropna()
    print(f"După dropna: {len(df):,} rânduri")
    return df


def standardize(df):
    """Z-score standardizare (zero mean, unit variance)."""
    A = df.values.astype(np.float64)
    m, n = A.shape
    means = [sum(float(A[i, j]) for i in range(m)) / m for j in range(n)]
    stds = []
    for j in range(n):
        var = sum((float(A[i, j]) - means[j]) ** 2 for i in range(m)) / m
        stds.append(math.sqrt(var) if var > 0 else 1.0)
    # evităm împărțirea la 0
    stds = [s if s > 0 else 1.0 for s in stds]
    for i in range(m):
        for j in range(n):
            A[i, j] = (A[i, j] - means[j]) / stds[j]
    return A


# 2. SVD MANUAL — folosind AᵀA eigen-decomposition


def jacobi_eigen_decomposition(M, max_iter: int = 100, tol: float = 1e-10):
    """
    Metoda Jacobi pentru descompunerea unei matrice simetrice în valori proprii.
    Returnează (eigvals, eigvecs) unde coloanele lui eigvecs sunt vectorii proprii.
    """
    n = M.shape[0]
    A = M.copy()
    V = np.zeros((n, n))
    for i in range(n):
        V[i, i] = 1.0

    for _ in range(max_iter):
        # Găsim cel mai mare element off-diagonal
        p, q = 0, 1
        max_val = abs(A[0, 1])
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j

        if max_val < tol:
            break

        # Calculăm unghiul de rotație
        if abs(A[p, p] - A[q, q]) < 1e-12:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(2.0 * A[p, q], A[q, q] - A[p, p])

        c = math.cos(theta)
        s = math.sin(theta)

        # Salvăm valorile vechi
        app = A[p, p]
        aqq = A[q, q]
        apq = A[p, q]

        # Actualizăm elementele A[p,p], A[q,q], A[p,q]
        A[p, p] = c * c * app - 2 * c * s * apq + s * s * aqq
        A[q, q] = s * s * app + 2 * c * s * apq + c * c * aqq
        A[p, q] = A[q, p] = 0.0

        # Actualizăm restul elementelor pe coloanele p și q
        for i in range(n):
            if i != p and i != q:
                a_ip = A[i, p]
                a_iq = A[i, q]
                A[i, p] = A[p, i] = c * a_ip - s * a_iq
                A[i, q] = A[q, i] = s * a_ip + c * a_iq

        # Actualizăm matricea de vectori proprii V = V · J
        for i in range(n):
            v_ip = V[i, p]
            v_iq = V[i, q]
            V[i, p] = c * v_ip - s * v_iq
            V[i, q] = s * v_ip + c * v_iq

    eigvals = _diag_extract(A)
    return np.array(eigvals), V


def manual_svd(A, k: int = None):
    """
    Calculează SVD manual:
        A = U · Σ · Vᵀ

    Pași:
        1. M = Aᵀ · A
        2. Valori proprii (λ) și vectori proprii (V) pentru M — prin metoda Jacobi
        3. Valori singulare: σ = sqrt(λ)
        4. U = A · V · Σ⁻¹
    """
    m, n = A.shape
    if k is None:
        k = min(m, n)

    # Pasul 1: M = AᵀA
    M = A.T @ A  # (n × n)

    # Pasul 2: valori proprii și vectori proprii (manual, metoda Jacobi)
    eigvals, eigvecs = jacobi_eigen_decomposition(M)

    # Sortăm descrescător după valori proprii
    idx = _argsort_desc(eigvals)
    eigvals = [eigvals[i] for i in idx]
    eigvecs = eigvecs[:, idx]

    # Pasul 3: V și Σ
    # Vectorii proprii sunt coloanele lui V
    V = eigvecs[:, :k]

    # Valori singulare (doar cele pozitive)
    singular_values = [math.sqrt(max(v, 0.0)) for v in eigvals[:k]]
    Sigma = _diag_matrix(singular_values)

    # Pasul 4: U = A · V · Σ⁻¹
    # Evităm împărțirea la 0 pentru σ≈0
    Sigma_inv = _diag_matrix([1.0 / (s + 1e-12) for s in singular_values])
    U = A @ V @ Sigma_inv

    # Păstrăm doar primele k coloane pentru U
    U = U[:, :k]

    return U, singular_values, V.T


def power_method_svd(A, k: int = 3, max_iter: int = 100, tol: float = 1e-10):
    """
    (Opțional) Power iteration pentru primele k componente singulare.
    Util pentru înțelegere, dar mai lent decât eigen-decomposition.
    """
    m, n = A.shape
    U_cols = []
    V_cols = []
    sigmas = []

    AtA = A.T @ A

    for _ in range(k):
        v = _randn(n)
        v = v / _norm(v)

        for __ in range(max_iter):
            v_new = AtA @ v
            v_new = v_new / _norm(v_new)
            if _norm(v_new - v) < tol:
                break
            v = v_new

        sigma = _norm(A @ v)
        u = (A @ v) / (sigma + 1e-12)

        V_cols.append(v)
        U_cols.append(u)
        sigmas.append(sigma)

        # Deflație: eliminăm componenta găsită
        A = A - sigma * _outer(u, v)
        AtA = A.T @ A

    # Construim matricile manual
    U = np.zeros((m, k))
    for j in range(k):
        U[:, j] = U_cols[j]
    Sigma = _diag_matrix(sigmas)
    Vt = np.zeros((k, n))
    for j in range(k):
        Vt[j, :] = V_cols[j]

    return U, Sigma, Vt


# 3. Interpretare conceptuală a grupurilor


def print_group_loadings(Vt, singular_values, feature_names):
    """
    Afișează contribuția fiecărei caracteristici în fiecare componentă (loadings).
    Loadings = V · Σ  (fiecare componentă înmulțită cu importanța sa).
    """
    # Loadings pentru fiecare caracteristică
    loadings = Vt.T * singular_values  # (n_features × n_components)

    groups = {
        "U — Preferințe piață": ["SALE PRICE", "NEIGHBORHOOD", "BUILDING CLASS CATEGORY"],
        "Σ — Importanță componente": ["TOTAL UNITS", "RESIDENTIAL UNITS", "COMMERCIAL UNITS"],
        "Vᵀ — Caracteristici proprietăți": [
            "GROSS SQUARE FEET",
            "LAND SQUARE FEET",
            "YEAR BUILT",
            "BUILDING CLASS AT PRESENT",
        ],
    }

    print("\n" + "=" * 60)
    print("INTERPRETAREA CONCEPTUALĂ A GRUPURILOR")
    print("=" * 60)

    for group_name, cols in groups.items():
        idx = [feature_names.index(c) for c in cols]
        group_loadings = loadings[idx, :]  # (group_size × n_components)

        print(f"\n📊 {group_name}")
        print("-" * 40)
        for i, col in enumerate(cols):
            contributions = group_loadings[i, :5]  # primele 5 componente
            contrib_str = " | ".join([f"PC{j+1}={v:+.3f}" for j, v in enumerate(contributions)])
            print(f"   {col:30s} → {contrib_str}")

    # Salvăm loadings în CSV
    loadings_df = pd.DataFrame(
        loadings[:, :5],
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(5)]
    )
    loadings_df.to_csv(os.path.join(RESULTS_DIR, "svd_loadings.csv"))
    print(f"\n✅ Loadings salvați în: {RESULTS_DIR}/svd_loadings.csv")


def print_variance_explained(singular_values):
    """Afișează varianța explicată de fiecare componentă."""
    variances = [s ** 2 for s in singular_values]
    total_var = sum(variances)
    explained = [v / total_var for v in variances]
    cumsum = _cumsum(explained)

    print("\n" + "=" * 60)
    print("VARIANȚA EXPLICATĂ DE FIECARE COMPONENTĂ")
    print("=" * 60)
    print(f"{'Componentă':>12} {'Singulară':>12} {'Varianță':>12} {'Cumulat':>12}")
    print("-" * 52)
    for i in range(min(6, len(singular_values))):
        print(f"PC{i+1:>11} {singular_values[i]:>12.4f} {explained[i]:>11.2%} {cumsum[i]:>11.2%}")

    # Salvăm
    var_df = pd.DataFrame({
        "singular_value": singular_values[:10],
        "variance_explained": explained[:10],
        "cumulative": cumsum[:10],
    })
    var_df.to_csv(os.path.join(RESULTS_DIR, "svd_variance_explained.csv"), index=False)
    print(f"\n✅ Varianța explicată salvată în: {RESULTS_DIR}/svd_variance_explained.csv")


# 4. Vizualizări


def plot_singular_values(singular_values):
    """Scree plot pentru valorile singulare."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(singular_values) + 1), singular_values, marker="o", color="steelblue")
    plt.title("Valori Singulare (Scree Plot)")
    plt.xlabel("Componentă")
    plt.ylabel("Valoare singulară (σ)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "12_svd_scree_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 12_svd_scree_plot.png")


def plot_loadings_heatmap(Vt, singular_values, feature_names):
    """Heatmap pentru loadings (primele 5 componente)."""
    loadings = Vt.T * singular_values
    loadings_df = pd.DataFrame(
        loadings[:, :5],
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(5)]
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        loadings_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
    )
    plt.title("SVD Loadings — Primele 5 Componente")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "13_svd_loadings_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 13_svd_loadings_heatmap.png")


def plot_biplot(A, U, Vt, feature_names):
    """
    Biplot 2D: proiecția datelor în spațiul primelor 2 componente singulare.
    """
    # Proiecția datelor în spațiul PC1/PC2
    # Z = A · V  →  coloanele V sunt vectorii proprii (nu Vt)
    V = Vt.T
    Z = A @ V[:, :2]

    # Pentru vizibilitate, sample 2000 puncte
    if len(Z) > 2000:
        idx = random.sample(range(len(Z)), 2000)
        Z = Z[idx]

    plt.figure(figsize=(10, 8))
    plt.scatter(Z[:, 0], Z[:, 1], alpha=0.4, s=15, color="steelblue")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("SVD Biplot — Proiecția proprietăților în spațiul PC1/PC2")
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "14_svd_biplot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 14_svd_biplot.png")


def plot_group_contributions(Vt, singular_values, feature_names):
    """
    Bar plot cu contribuția absolută a fiecărui grup de caracteristici
    pentru primele 3 componente.
    """
    loadings = np.abs(Vt.T * singular_values)  # (n_features × n_components)

    groups = {
        "U — Preferințe piață": ["SALE PRICE", "NEIGHBORHOOD", "BUILDING CLASS CATEGORY"],
        "Σ — Importanță": ["TOTAL UNITS", "RESIDENTIAL UNITS", "COMMERCIAL UNITS"],
        "Vᵀ — Caracteristici": [
            "GROSS SQUARE FEET",
            "LAND SQUARE FEET",
            "YEAR BUILT",
            "BUILDING CLASS AT PRESENT",
        ],
    }

    contributions = {}
    for name, cols in groups.items():
        idx = [feature_names.index(c) for c in cols]
        contributions[name] = loadings[idx, :3].sum(axis=0)  # suma pe primele 3 PC

    df_contrib = pd.DataFrame(contributions, index=[f"PC{i+1}" for i in range(3)]).T

    df_contrib.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black")
    plt.title("Contribuția grupurilor de caracteristici în primele 3 componente")
    plt.ylabel("Suma loading-urilor absolute")
    plt.xlabel("Grup conceptual")
    plt.xticks(rotation=15, ha="right")
    plt.legend(title="Componentă")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "15_svd_group_contributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 15_svd_group_contributions.png")


# 5. MAIN

def main():
    print("=" * 60)
    print("SVD MANUAL — NYC Real Estate")
    print("=" * 60)

    df = load_and_preprocess(INPUT_PATH)
    feature_names = list(df.columns)

    print("\nStandardizare matrice A...")
    A = standardize(df)
    print(f"Matrice A: {A.shape[0]:,} rânduri × {A.shape[1]} coloane")

    print("\nCalcul SVD manual (AᵀA eigen-decomposition)...")
    U, singular_values, Vt = manual_svd(A, k=min(A.shape))

    print(f"U shape:   {U.shape}")
    print(f"Σ shape:   ({len(singular_values)}, {len(singular_values)})")
    print(f"Vᵀ shape:  {Vt.shape}")

    # Verificare reconstrucție (norma Frobenius calculată manual)
    A_reconstructed = U @ _diag_matrix(singular_values) @ Vt
    diff = A - A_reconstructed
    fro_diff = math.sqrt(sum(float(diff[i, j]) ** 2 for i in range(diff.shape[0]) for j in range(diff.shape[1])))
    fro_orig = math.sqrt(sum(float(A[i, j]) ** 2 for i in range(A.shape[0]) for j in range(A.shape[1])))
    reconstr_error = fro_diff / fro_orig
    print(f"\nEroare reconstrucție (relativă, Frobenius): {reconstr_error:.2e}")

    # Interpretare
    print_variance_explained(singular_values)
    print_group_loadings(Vt, singular_values, feature_names)

    # Vizualizări
    print("\nGenerare grafice SVD...")
    plot_singular_values(singular_values)
    plot_loadings_heatmap(Vt, singular_values, feature_names)
    plot_biplot(A, U, Vt, feature_names)
    plot_group_contributions(Vt, singular_values, feature_names)

    # Salvare rezultate brute (CSV în loc de npy)
    pd.DataFrame(U[:, :5]).to_csv(os.path.join(RESULTS_DIR, "svd_U.csv"), index=False)
    pd.DataFrame(_diag_matrix(singular_values[:5])).to_csv(os.path.join(RESULTS_DIR, "svd_Sigma.csv"), index=False)
    pd.DataFrame(Vt[:5, :]).to_csv(os.path.join(RESULTS_DIR, "svd_Vt.csv"), index=False)
    print(f"\n✅ Matricile U, Σ, Vᵀ (primele 5 componente) salvate în {RESULTS_DIR}/")

    print("\n" + "=" * 60)
    print("SVD MANUAL finalizat cu succes!")
    print("=" * 60)


if __name__ == "__main__":
    main()
