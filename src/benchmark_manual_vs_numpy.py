#!/usr/bin/env python3
"""
benchmark_manual_vs_numpy.py

Comparație sistematică între funcțiile implementate manual în proiect
și echivalentele predefinite din NumPy / SciPy / scikit-learn.

Măsoară:
  • Timp de execuție (secunde)
  • Memorie maximă utilizată (MB)
  • Acuratețe numerică (eroare relativă față de referința NumPy)

Rezultatele sunt salvate în results/ și charts/.
"""

import os
import sys
import time
import tracemalloc
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import funcții manuale
# ---------------------------------------------------------------------------
from manual_math import mean, percentile, argsort, unique
from custom_svd import SVD as manual_svd_full, _norm as manual_norm, _outer as manual_outer
from svd_manual import standardize as manual_standardize, _cumsum as manual_cumsum

# ---------------------------------------------------------------------------
# Configurare căi
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
CHARTS_DIR = "charts"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Utilitare de măsurare
# ---------------------------------------------------------------------------


def measure(func, *args, repetitions=3, warmup=1, **kwargs):
    """
    Măsoară timpul minim de execuție și memoria peak (via tracemalloc)
    pentru func(*args, **kwargs).
    Returnează dict cu: time_min, time_mean, memory_peak_mb, result.
    """
    # Warmup
    for _ in range(warmup):
        try:
            func(*args, **kwargs)
        except Exception:
            pass

    times = []
    for _ in range(repetitions):
        tracemalloc.start()
        tracemalloc.reset_peak()
        t0 = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            tracemalloc.stop()
            raise e
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(t1 - t0)

    return {
        "time_min": min(times),
        "time_mean": sum(times) / len(times),
        "memory_peak_mb": peak / (1024 * 1024),
        "result": result,
    }


def relative_error(manual_res, numpy_res, atol=1e-8):
    """
    Calculează eroarea relativă / absolută între două rezultate.
    Suportă scalari, liste și numpy arrays.
    """
    try:
        a = np.asarray(manual_res, dtype=np.float64).ravel()
        b = np.asarray(numpy_res, dtype=np.float64).ravel()
        if a.shape != b.shape:
            return np.inf
        diff = np.linalg.norm(a - b)
        norm_b = np.linalg.norm(b)
        if norm_b < atol:
            return diff
        return diff / norm_b
    except Exception:
        if isinstance(manual_res, (list, tuple)) and isinstance(numpy_res, (list, tuple)):
            if len(manual_res) != len(numpy_res):
                return np.inf
            diffs = []
            for x, y in zip(manual_res, numpy_res):
                if abs(y) < atol:
                    diffs.append(abs(x - y))
                else:
                    diffs.append(abs(x - y) / abs(y))
            return max(diffs)
        return abs(float(manual_res) - float(numpy_res)) / max(abs(float(numpy_res)), atol)


# ---------------------------------------------------------------------------
# Benchmark-uri individuale
# ---------------------------------------------------------------------------


def benchmark_mean(size):
    data_list = list(np.random.randn(size))
    data_arr = np.array(data_list)

    m_manual = measure(mean, data_list, repetitions=5)
    m_numpy = measure(np.mean, data_arr, repetitions=5)

    return {
        "func": "mean",
        "size": size,
        "manual_time": m_manual["time_min"],
        "numpy_time": m_numpy["time_min"],
        "manual_mem_mb": m_manual["memory_peak_mb"],
        "numpy_mem_mb": m_numpy["memory_peak_mb"],
        "rel_error": relative_error(m_manual["result"], m_numpy["result"]),
    }


def benchmark_percentile(size):
    data_list = list(np.random.randn(size))
    data_arr = np.array(data_list)
    p = 75

    m_manual = measure(percentile, data_list, p, repetitions=5)
    m_numpy = measure(np.percentile, data_arr, p, method="linear", repetitions=5)

    return {
        "func": "percentile_75",
        "size": size,
        "manual_time": m_manual["time_min"],
        "numpy_time": m_numpy["time_min"],
        "manual_mem_mb": m_manual["memory_peak_mb"],
        "numpy_mem_mb": m_numpy["memory_peak_mb"],
        "rel_error": relative_error(m_manual["result"], m_numpy["result"]),
    }


def benchmark_argsort(size):
    data_list = list(np.random.randn(size))
    data_arr = np.array(data_list)

    m_manual = measure(argsort, data_list, repetitions=5)
    m_numpy = measure(np.argsort, data_arr, repetitions=5)

    manual_res = np.array(m_manual["result"])
    numpy_res = m_numpy["result"]

    return {
        "func": "argsort",
        "size": size,
        "manual_time": m_manual["time_min"],
        "numpy_time": m_numpy["time_min"],
        "manual_mem_mb": m_manual["memory_peak_mb"],
        "numpy_mem_mb": m_numpy["memory_peak_mb"],
        "rel_error": relative_error(manual_res, numpy_res),
    }


def benchmark_unique(size):
    data_list = list(np.random.randint(0, size // 2, size=size))
    data_arr = np.array(data_list)

    m_manual = measure(unique, data_list, repetitions=5)
    m_numpy = measure(np.unique, data_arr, repetitions=5)

    manual_res = np.array(m_manual["result"])
    numpy_res = m_numpy["result"]

    return {
        "func": "unique",
        "size": size,
        "manual_time": m_manual["time_min"],
        "numpy_time": m_numpy["time_min"],
        "manual_mem_mb": m_manual["memory_peak_mb"],
        "numpy_mem_mb": m_numpy["memory_peak_mb"],
        "rel_error": relative_error(manual_res, numpy_res),
    }


def benchmark_norm(size):
    data_list = list(np.random.randn(size))
    data_arr = np.array(data_list)

    m_manual = measure(manual_norm, data_list, repetitions=5)
    m_numpy = measure(np.linalg.norm, data_arr, repetitions=5)

    return {
        "func": "norm_L2",
        "size": size,
        "manual_time": m_manual["time_min"],
        "numpy_time": m_numpy["time_min"],
        "manual_mem_mb": m_manual["memory_peak_mb"],
        "numpy_mem_mb": m_numpy["memory_peak_mb"],
        "rel_error": relative_error(m_manual["result"], m_numpy["result"]),
    }


def benchmark_outer(size):
    v = list(np.random.randn(size))
    w = list(np.random.randn(size))
    v_arr = np.array(v)
    w_arr = np.array(w)

    m_manual = measure(manual_outer, v, w, repetitions=3)
    m_numpy = measure(np.outer, v_arr, w_arr, repetitions=3)

    return {
        "func": "outer_product",
        "size": size,
        "manual_time": m_manual["time_min"],
        "numpy_time": m_numpy["time_min"],
        "manual_mem_mb": m_manual["memory_peak_mb"],
        "numpy_mem_mb": m_numpy["memory_peak_mb"],
        "rel_error": relative_error(m_manual["result"], m_numpy["result"]),
    }


def benchmark_cumsum(size):
    data_list = list(np.random.randn(size))
    data_arr = np.array(data_list)

    m_manual = measure(manual_cumsum, data_list, repetitions=5)
    m_numpy = measure(np.cumsum, data_arr, repetitions=5)

    return {
        "func": "cumsum",
        "size": size,
        "manual_time": m_manual["time_min"],
        "numpy_time": m_numpy["time_min"],
        "manual_mem_mb": m_manual["memory_peak_mb"],
        "numpy_mem_mb": m_numpy["memory_peak_mb"],
        "rel_error": relative_error(m_manual["result"], m_numpy["result"]),
    }


def benchmark_standardize(rows, cols):
    df_vals = np.random.randn(rows, cols)
    df = pd.DataFrame(df_vals, columns=[f"c{i}" for i in range(cols)])

    def run_manual():
        return manual_standardize(df.copy())

    def run_numpy():
        X = df_vals.copy()
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1.0
        return (X - means) / stds

    m_manual = measure(run_manual, repetitions=3)
    m_numpy = measure(run_numpy, repetitions=3)

    return {
        "func": "standardize",
        "size": f"{rows}x{cols}",
        "manual_time": m_manual["time_min"],
        "numpy_time": m_numpy["time_min"],
        "manual_mem_mb": m_manual["memory_peak_mb"],
        "numpy_mem_mb": m_numpy["memory_peak_mb"],
        "rel_error": relative_error(m_manual["result"], m_numpy["result"]),
    }


def benchmark_svd(n):
    A = np.random.randn(n, n)

    def run_manual():
        return manual_svd_full(A.copy())

    def run_numpy():
        return np.linalg.svd(A.copy(), full_matrices=False)

    rep_numpy = 5
    rep_manual = 1 if n > 80 else 3

    m_manual = measure(run_manual, repetitions=rep_manual)
    m_numpy = measure(run_numpy, repetitions=rep_numpy)

    U_m, S_m, V_m = m_manual["result"]
    U_n, s_n, Vt_n = m_numpy["result"]

    A_rec_manual = U_m @ S_m @ V_m.T
    A_rec_numpy = U_n @ np.diag(s_n) @ Vt_n

    return {
        "func": "svd",
        "size": f"{n}x{n}",
        "manual_time": m_manual["time_min"],
        "numpy_time": m_numpy["time_min"],
        "manual_mem_mb": m_manual["memory_peak_mb"],
        "numpy_mem_mb": m_numpy["memory_peak_mb"],
        "rel_error": relative_error(A_rec_manual, A_rec_numpy),
    }


def benchmark_reconstruction_error(rows, cols):
    X = np.random.randn(rows, cols)
    X_rec = np.random.randn(rows, cols)

    def manual_loop():
        m, n = len(X), X.shape[1]
        res = []
        for i in range(m):
            s = 0.0
            for j in range(n):
                diff = float(X[i, j]) - float(X_rec[i, j])
                s += diff * diff
            res.append(s)
        return np.array(res)

    def numpy_vec():
        return np.sum((X - X_rec) ** 2, axis=1)

    m_manual = measure(manual_loop, repetitions=3)
    m_numpy = measure(numpy_vec, repetitions=5)

    return {
        "func": "reconstruction_error",
        "size": f"{rows}x{cols}",
        "manual_time": m_manual["time_min"],
        "numpy_time": m_numpy["time_min"],
        "manual_mem_mb": m_manual["memory_peak_mb"],
        "numpy_mem_mb": m_numpy["memory_peak_mb"],
        "rel_error": relative_error(m_manual["result"], m_numpy["result"]),
    }


def benchmark_euclidean_distance(rows, cols):
    X = np.random.randn(rows, cols)
    centroids = np.random.randn(4, cols)
    labels = np.random.randint(0, 4, size=rows)

    def manual_loop():
        m, n = len(X), X.shape[1]
        dists = []
        for i in range(m):
            c = centroids[labels[i]]
            s = 0.0
            for j in range(n):
                diff = float(X[i, j]) - float(c[j])
                s += diff * diff
            dists.append(math.sqrt(s))
        return np.array(dists)

    def numpy_vec():
        return np.linalg.norm(X - centroids[labels], axis=1)

    m_manual = measure(manual_loop, repetitions=3)
    m_numpy = measure(numpy_vec, repetitions=5)

    return {
        "func": "euclidean_distance",
        "size": f"{rows}x{cols}",
        "manual_time": m_manual["time_min"],
        "numpy_time": m_numpy["time_min"],
        "manual_mem_mb": m_manual["memory_peak_mb"],
        "numpy_mem_mb": m_numpy["memory_peak_mb"],
        "rel_error": relative_error(m_manual["result"], m_numpy["result"]),
    }


# ---------------------------------------------------------------------------
# Orchestrare benchmark-uri
# ---------------------------------------------------------------------------

def run_all_benchmarks():
    results = []

    print("=" * 70)
    print("BENCHMARK: Functii manuale vs NumPy")
    print("=" * 70)

    # 1. Functii simple pe vectori
    sizes_vector = [1_000, 10_000, 100_000]
    vector_benches = [
        benchmark_mean,
        benchmark_percentile,
        benchmark_argsort,
        benchmark_unique,
        benchmark_norm,
        benchmark_cumsum,
    ]
    for size in sizes_vector:
        print(f"\n[Vector size={size:,}]")
        for bench_fn in vector_benches:
            label = bench_fn.__name__.replace("benchmark_", "")
            print(f"  Running {label} ...", end=" ")
            try:
                res = bench_fn(size)
                results.append(res)
                print(f"OK (manual={res['manual_time']:.4f}s, numpy={res['numpy_time']:.4f}s)")
            except Exception as e:
                print(f"FAIL: {e}")

    # 2. Outer product (memorie O(n^2))
    outer_sizes = [100, 500, 1000]
    for size in outer_sizes:
        print(f"\n[Outer size={size}]")
        try:
            res = benchmark_outer(size)
            results.append(res)
            print(f"  OK (manual={res['manual_time']:.4f}s, numpy={res['numpy_time']:.4f}s)")
        except Exception as e:
            print(f"  FAIL: {e}")

    # 3. Standardizare
    std_sizes = [(1_000, 10), (10_000, 10), (50_000, 10)]
    for rows, cols in std_sizes:
        print(f"\n[Standardize {rows}x{cols}]")
        try:
            res = benchmark_standardize(rows, cols)
            results.append(res)
            print(f"  OK (manual={res['manual_time']:.4f}s, numpy={res['numpy_time']:.4f}s)")
        except Exception as e:
            print(f"  FAIL: {e}")

    # 4. SVD (dimensiuni mici, manualul e foarte lent)
    svd_sizes = [20, 50, 100]
    for n in svd_sizes:
        print(f"\n[SVD {n}x{n}]")
        try:
            res = benchmark_svd(n)
            results.append(res)
            print(f"  OK (manual={res['manual_time']:.4f}s, numpy={res['numpy_time']:.4f}s)")
        except Exception as e:
            print(f"  FAIL: {e}")

    # 5. Reconstruction error
    rec_sizes = [(1_000, 10), (10_000, 10), (50_000, 10)]
    for rows, cols in rec_sizes:
        print(f"\n[Reconstruction error {rows}x{cols}]")
        try:
            res = benchmark_reconstruction_error(rows, cols)
            results.append(res)
            print(f"  OK (manual={res['manual_time']:.4f}s, numpy={res['numpy_time']:.4f}s)")
        except Exception as e:
            print(f"  FAIL: {e}")

    # 6. Euclidean distance
    euc_sizes = [(1_000, 10), (10_000, 10), (50_000, 10)]
    for rows, cols in euc_sizes:
        print(f"\n[Euclidean distance {rows}x{cols}]")
        try:
            res = benchmark_euclidean_distance(rows, cols)
            results.append(res)
            print(f"  OK (manual={res['manual_time']:.4f}s, numpy={res['numpy_time']:.4f}s)")
        except Exception as e:
            print(f"  FAIL: {e}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Vizualizare
# ---------------------------------------------------------------------------

def plot_results(df):
    # Coloane derivate
    df["speedup"] = df["manual_time"] / df["numpy_time"]
    df["speedup"] = df["speedup"].replace([np.inf, -np.inf], np.nan)

    # Salvare CSV
    csv_path = os.path.join(RESULTS_DIR, "benchmark_manual_vs_numpy.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRezultate salvate in: {csv_path}")

    # Tabel consola
    print("\n" + "=" * 100)
    print("REZUMAT BENCHMARK")
    print("=" * 100)
    display_cols = ["func", "size", "manual_time", "numpy_time", "speedup", "rel_error"]
    print(df[display_cols].to_string(index=False))

    # -----------------------------------------------------------------------
    # Figura cu 4 subploturi
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Agregare medii pe functie
    agg = df.groupby("func").agg({"manual_time": "mean", "numpy_time": "mean"}).reset_index()
    agg = agg.sort_values("manual_time", ascending=False)

    # --- Subplot 1: Timp de executie ---
    ax = axes[0, 0]
    ax.barh(agg["func"], agg["manual_time"], color="steelblue", label="Manual (loop Python)")
    ax.barh(agg["func"], agg["numpy_time"], left=0, color="coral", label="NumPy (C vectorizat)", alpha=0.8)
    ax.set_xlabel("Timp mediu in secunde (scara logaritmica)")
    ax.set_xscale("log")
    ax.set_title("Timp de executie per functie\nAlbastru = Manual   |   Portocaliu = NumPy", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # --- Subplot 2: Speedup ---
    ax = axes[0, 1]
    agg_speedup = df.groupby("func").agg({"speedup": "mean"}).reset_index()
    agg_speedup = agg_speedup.sort_values("speedup", ascending=False)
    colors = ["#2ecc71" if s > 1 else "#e74c3c" for s in agg_speedup["speedup"]]
    bars = ax.barh(agg_speedup["func"], agg_speedup["speedup"], color=colors)
    ax.axvline(1, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Raport de viteza (scara logaritmica)")
    ax.set_xscale("log")
    ax.set_title("Cat de multe ori e mai rapid NumPy?\nSpeedup = Timp Manual / Timp NumPy", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Explicatie speedup
    ax.text(
        0.97, 0.05,
        "O bara la 100 inseamna:\nNumPy e de 100x mai rapid",
        transform=ax.transAxes, fontsize=10, verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.9)
    )

    # Valori pe bare
    for bar, val in zip(bars, agg_speedup["speedup"]):
        if not np.isnan(val):
            ax.text(val * 1.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}x", va="center", ha="left", fontsize=8, fontweight="bold")

    # --- Subplot 3: Eroare relativa ---
    ax = axes[1, 0]
    agg_err = df.groupby("func").agg({"rel_error": "max"}).reset_index()
    agg_err = agg_err.sort_values("rel_error", ascending=False)
    bars = ax.barh(agg_err["func"], agg_err["rel_error"], color="mediumpurple")
    ax.set_xlabel("Eroare relativa maxima (scara logaritmica)")
    ax.set_title("Diferenta numerica: Manual vs NumPy\nEroare = |Manual - NumPy| / |NumPy|\nNumPy = referinta (adevarul)", fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Linie precizie masina float64
    ax.axvline(2.2e-16, color="red", linestyle=":", linewidth=1.5, label="Precizie masina float64 (~2.2e-16)")
    ax.legend(loc="lower right")

    # Explicatie eroare (sus-dreapta, fara suprapunere)
    ax.text(
        0.97, 0.95,
        "1e-15 = rezultate identice\npana la a 15-a zecimala",
        transform=ax.transAxes, fontsize=10, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9)
    )

    # Valori pe bare
    for bar, val in zip(bars, agg_err["rel_error"]):
        if val > 0:
            ax.text(val * 1.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1e}", va="center", ha="left", fontsize=8, fontweight="bold")

    # --- Subplot 4: Memorie peak ---
    ax = axes[1, 1]
    agg_mem = df.groupby("func").agg({"manual_mem_mb": "mean", "numpy_mem_mb": "mean"}).reset_index()
    agg_mem = agg_mem.sort_values("manual_mem_mb", ascending=False)
    x = np.arange(len(agg_mem))
    width = 0.35
    ax.bar(x - width / 2, agg_mem["manual_mem_mb"], width, label="Manual (loop)", color="steelblue")
    ax.bar(x + width / 2, agg_mem["numpy_mem_mb"], width, label="NumPy (vectorizat)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(agg_mem["func"], rotation=45, ha="right")
    ax.set_ylabel("Memorie maxima folosita (MB)")
    ax.set_title("Memorie maxima (peak) in timpul rularii\nAlbastru = Manual   |   Portocaliu = NumPy", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Legenda generala a figurii
    fig.text(
        0.5, 0.005,
        "ALBASTRU = Implementare manuala cu loop-uri Python  |  PORTOCALIU = NumPy predefinit (C/Fortran/BLAS)",
        ha="center", fontsize=11, fontweight="bold", color="darkslategray",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3)
    )

    plt.suptitle("Benchmark: Functii manuale vs NumPy -- Timp · Viteza · Acuratete · Memorie", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    chart_path = os.path.join(CHARTS_DIR, "17_benchmark_manual_vs_numpy.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafic salvat in: {chart_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = run_all_benchmarks()
    plot_results(df)
    print("\n=== Benchmark finalizat cu succes! ===")
