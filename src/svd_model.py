import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from custom_svd import SVD

RESULTS_DIR = "results"

A = pd.read_csv(os.path.join(RESULTS_DIR, "matrix_A.csv")).values
y = pd.read_csv(os.path.join(RESULTS_DIR, "vector_y.csv")).values.flatten()

U, S, V = SVD(A)

sigma_inv = np.zeros((S.shape[1], S.shape[0]))

for i in range(min(S.shape)):
    if S[i, i] > 1e-10:
        sigma_inv[i, i] = 1 / S[i, i]

A_pinv = V @ sigma_inv @ U.T

x = A_pinv @ y

y_pred = A @ x

mae = np.mean(np.abs(y - y_pred))
rmse = np.sqrt(np.mean((y - y_pred) ** 2))

print("MAE:", mae)
print("RMSE:", rmse)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Subplot 1: Zoomed linear view (bulk of data) ---
mask_zoom = (y < 50_000_000) & (y_pred > -10_000_000) & (y_pred < 50_000_000)
axes[0].scatter(y[mask_zoom], y_pred[mask_zoom], alpha=0.35, s=15, edgecolors='none')
axes[0].plot([0, 50_000_000], [0, 50_000_000], 'r--', lw=1.5, label='Perfect prediction (y=x)')
axes[0].set_xlim(-2_000_000, 50_000_000)
axes[0].set_ylim(-10_000_000, 50_000_000)
axes[0].set_xlabel("Real Sale Price ($)")
axes[0].set_ylabel("Predicted Sale Price ($)")
axes[0].set_title(f"Zoomed View (< $50M)\n{mask_zoom.sum():,} / {len(y):,} properties")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].ticklabel_format(style='plain', axis='both')

# --- Subplot 2: Log-log view (positive values only) ---
mask_log = (y > 0) & (y_pred > 0)
axes[1].scatter(y[mask_log], y_pred[mask_log], alpha=0.35, s=15, edgecolors='none')
min_val = min(y[mask_log].min(), y_pred[mask_log].min())
max_val = max(y[mask_log].max(), y_pred[mask_log].max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Perfect prediction (y=x)')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel("Real Sale Price ($) — log scale")
axes[1].set_ylabel("Predicted Sale Price ($) — log scale")
axes[1].set_title(f"Log-Log View (positive only)\n{mask_log.sum():,} / {len(y):,} properties")
axes[1].legend()
axes[1].grid(True, alpha=0.3, which='both')

plt.suptitle("Real vs Predicted Sale Price — SVD Least Squares (Full Dataset)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "real_vs_predicted_svd.png"), dpi=150)
plt.close()

print("Saved: results/real_vs_predicted_svd.png")
