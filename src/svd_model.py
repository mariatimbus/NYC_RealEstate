import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from custom_svd import SVD

RESULTS_DIR = "results"

A = pd.read_csv(os.path.join(RESULTS_DIR, "matrix_A.csv")).values
y = pd.read_csv(os.path.join(RESULTS_DIR, "vector_y.csv")).values.flatten()

A = A[:100, :]
y = y[:100]
A = A[:, :30]

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

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.4)
plt.xlabel("Real Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Real vs Predicted Sale Price")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "real_vs_predicted_svd.png"))
plt.close()

print("Saved: results/real_vs_predicted_svd.png")