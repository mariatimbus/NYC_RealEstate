import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

INPUT_PATH = "data/cleaned_dataset.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(INPUT_PATH)

# Target vector
y = df["SALE PRICE"]

# Feature matrix
X = df.drop(columns=["SALE PRICE"])

# One-hot encoding for categorical variables
X = pd.get_dummies(
    X,
    columns=[
        "NEIGHBORHOOD",
        "BUILDING CLASS CATEGORY",
        "BUILDING CLASS AT PRESENT"
    ]
)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save matrix and target
pd.DataFrame(X_scaled).to_csv(
    os.path.join(RESULTS_DIR, "matrix_A.csv"),
    index=False
)

pd.DataFrame(y).to_csv(
    os.path.join(RESULTS_DIR, "vector_y.csv"),
    index=False
)

print("Matrix A shape:", X_scaled.shape)
print("Vector y shape:", y.shape)

print("\nMatrix A and vector y saved successfully!")