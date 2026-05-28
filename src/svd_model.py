import os
import pandas as pd
import matplotlib.pyplot as plt

from custom_svd import SVD

RESULTS_DIR = "results"

A = pd.read_csv(os.path.join(RESULTS_DIR, "matrix_A.csv")).values
y = pd.read_csv(os.path.join(RESULTS_DIR, "vector_y.csv")).values.flatten()

U, S, V = SVD(A)

m = S.shape[0]
n = S.shape[1]
sigma_inv = [[0.0] * m for _ in range(n)]

for i in range(min(m, n)):
    if S[i, i] > 1e-10:
        sigma_inv[i][i] = 1.0 / S[i, i]

sigma_inv = pd.DataFrame(sigma_inv).values
A_pinv = V @ sigma_inv @ U.T

x = A_pinv @ y

y_pred = A @ x

# MAE — manual
mae = sum(abs(float(y[i]) - float(y_pred[i])) for i in range(len(y))) / len(y)

# RMSE — manual
rmse = (sum((float(y[i]) - float(y_pred[i])) ** 2 for i in range(len(y))) / len(y)) ** 0.5

print("MAE:", mae)
print("RMSE:", rmse)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Subplot 1: Zoomed linear view (bulk of data) ---
mask_zoom = [(y[i] < 50_000_000) and (y_pred[i] > -10_000_000) and (y_pred[i] < 50_000_000) for i in range(len(y))]
y_zoom = [float(y[i]) for i in range(len(y)) if mask_zoom[i]]
pred_zoom = [float(y_pred[i]) for i in range(len(y)) if mask_zoom[i]]
axes[0].scatter(y_zoom, pred_zoom, alpha=0.35, s=15, edgecolors='none')
axes[0].plot([0, 50_000_000], [0, 50_000_000], 'r--', lw=1.5, label='Perfect prediction (y=x)')
axes[0].set_xlim(-2_000_000, 50_000_000)
axes[0].set_ylim(-10_000_000, 50_000_000)
axes[0].set_xlabel("Real Sale Price ($)")
axes[0].set_ylabel("Predicted Sale Price ($)")
axes[0].set_title(f"Zoomed View (< $50M)\n{sum(mask_zoom):,} / {len(y):,} properties")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].ticklabel_format(style='plain', axis='both')

# --- Subplot 2: Log-log view (positive values only) ---
mask_log = [(y[i] > 0) and (y_pred[i] > 0) for i in range(len(y))]
y_log_vals = [float(y[i]) for i in range(len(y)) if mask_log[i]]
pred_log_vals = [float(y_pred[i]) for i in range(len(y)) if mask_log[i]]
axes[1].scatter(y_log_vals, pred_log_vals, alpha=0.35, s=15, edgecolors='none')
min_val = min(min(y_log_vals), min(pred_log_vals))
max_val = max(max(y_log_vals), max(pred_log_vals))
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Perfect prediction (y=x)')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel("Real Sale Price ($) — log scale")
axes[1].set_ylabel("Predicted Sale Price ($) — log scale")
axes[1].set_title(f"Log-Log View (positive only)\n{sum(mask_log):,} / {len(y):,} properties")
axes[1].legend()
axes[1].grid(True, alpha=0.3, which='both')

plt.suptitle("Real vs Predicted Sale Price — SVD Least Squares (Full Dataset)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "real_vs_predicted_svd.png"), dpi=150)
plt.close()

print("Saved: results/real_vs_predicted_svd.png")
