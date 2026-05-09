#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
GRAPHS_DIR = "results/graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

BLUE = "#2563EB"
GREEN = "#10B981"
ORANGE = "#F59E0B"
PURPLE = "#8B5CF6"
RED = "#EF4444"
TEAL = "#14B8A6"
PINK = "#EC4899"

sns.set_theme(style="whitegrid")


def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, name))
    plt.close()


# 1. Building classes
df = pd.read_csv("results/building_class_stats.csv")
top = df.sort_values("median_price", ascending=False).head(15)

plt.figure(figsize=(13, 7))
plt.bar(top["BUILDING CLASS CATEGORY"], top["median_price"], color=PURPLE)
plt.title("Top 15 Building Classes by Median Sale Price", fontsize=18, fontweight="bold")
plt.xlabel("Building Class Category")
plt.ylabel("Median Sale Price")
plt.xticks(rotation=75, ha="right")
plt.grid(axis="y", alpha=0.3)
save_plot("building_class_median_price.png")


# 2. Neighborhoods
df = pd.read_csv("results/neighborhood_stats.csv")
top = df.sort_values("median_price", ascending=False).head(15)

plt.figure(figsize=(13, 7))
plt.bar(top["NEIGHBORHOOD"], top["median_price"], color=BLUE)
plt.title("Top 15 Neighborhoods by Median Sale Price", fontsize=18, fontweight="bold")
plt.xlabel("Neighborhood")
plt.ylabel("Median Sale Price")
plt.xticks(rotation=75, ha="right")
plt.grid(axis="y", alpha=0.3)
save_plot("neighborhood_median_price.png")


# 3. K-Means elbow
df = pd.read_csv("results/kmeans_inertia.csv")

plt.figure(figsize=(10, 6))
plt.plot(df["k"], df["inertia"], marker="o", color=ORANGE, linewidth=2.5)
plt.title("K-Means Elbow Method", fontsize=18, fontweight="bold")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True, alpha=0.3)
save_plot("kmeans_elbow.png")


# 4. Cluster profiles
df = pd.read_csv("results/cluster_profiles.csv")

plt.figure(figsize=(10, 6))
plt.bar(df["CLUSTER"].astype(str), df["count"], color=GREEN)
plt.title("Number of Properties per Cluster", fontsize=18, fontweight="bold")
plt.xlabel("Cluster")
plt.ylabel("Number of Properties")
plt.grid(axis="y", alpha=0.3)
save_plot("cluster_distribution.png")

plt.figure(figsize=(10, 6))
plt.bar(df["CLUSTER"].astype(str), df["median_price"], color=TEAL)
plt.title("Median Sale Price per Cluster", fontsize=18, fontweight="bold")
plt.xlabel("Cluster")
plt.ylabel("Median Sale Price")
plt.grid(axis="y", alpha=0.3)
save_plot("cluster_median_price.png")


# 5. Descriptive statistics
df = pd.read_csv("results/descriptive_statistics.csv")
df = df.rename(columns={"Unnamed: 0": "feature"})

plt.figure(figsize=(12, 6))
plt.bar(df["feature"], df["median"], color=PINK)
plt.title("Median Values of Numeric Features", fontsize=18, fontweight="bold")
plt.xlabel("Feature")
plt.ylabel("Median Value")
plt.xticks(rotation=60, ha="right")
plt.grid(axis="y", alpha=0.3)
save_plot("descriptive_statistics_median.png")


# 6. Correlation matrix
df = pd.read_csv("results/correlation_matrix.csv")
df = df.rename(columns={"Unnamed: 0": "feature"})

corr = df[["feature", "SALE PRICE"]]
corr = corr[corr["feature"] != "SALE PRICE"]
corr = corr.sort_values("SALE PRICE", ascending=False)

corr_colors = [GREEN if value > 0 else RED for value in corr["SALE PRICE"]]

plt.figure(figsize=(12, 6))
plt.bar(corr["feature"], corr["SALE PRICE"], color=corr_colors)
plt.title("Correlation with Sale Price", fontsize=18, fontweight="bold")
plt.xlabel("Feature")
plt.ylabel("Correlation")
plt.xticks(rotation=60, ha="right")
plt.grid(axis="y", alpha=0.3)
save_plot("correlation_with_sale_price.png")


# 7. Price per square foot
df = pd.read_csv("results/price_per_sqft_stats.csv")
df = df.rename(columns={"Unnamed: 0": "statistic"})

median_row = df[df["statistic"] == "50%"]

plt.figure(figsize=(8, 6))
plt.bar(
    ["Gross sqft", "Land sqft"],
    [
        median_row["PRICE_PER_GROSS_SQFT"].values[0],
        median_row["PRICE_PER_LAND_SQFT"].values[0],
    ],
    color=[BLUE, ORANGE]
)
plt.title("Median Price per Square Foot", fontsize=18, fontweight="bold")
plt.ylabel("Price per Square Foot")
plt.grid(axis="y", alpha=0.3)
save_plot("price_per_sqft_median.png")


# 8. Price vs surface by cluster
df = pd.read_csv("results/clustered_dataset.csv")

filtered = df[
    (df["SALE PRICE"] <= df["SALE PRICE"].quantile(0.99)) &
    (df["GROSS SQUARE FEET"] <= df["GROSS SQUARE FEET"].quantile(0.99))
]

plt.figure(figsize=(11, 7))

clusters = sorted(filtered["CLUSTER"].unique())
cluster_colors = [BLUE, ORANGE, GREEN, PURPLE, RED]

for i, cluster in enumerate(clusters):
    cluster_data = filtered[filtered["CLUSTER"] == cluster]

    plt.scatter(
        cluster_data["GROSS SQUARE FEET"],
        cluster_data["SALE PRICE"],
        alpha=0.5,
        s=35,
        color=cluster_colors[i % len(cluster_colors)],
        label=f"Cluster {cluster}"
    )

plt.title("Sale Price vs Gross Square Feet by Cluster", fontsize=18, fontweight="bold")
plt.xlabel("Gross Square Feet")
plt.ylabel("Sale Price")
plt.legend(title="K-Means Clusters")
plt.grid(True, alpha=0.3)
save_plot("price_vs_surface_by_cluster.png")


# 9. Price distribution
filtered_prices = df[
    df["SALE PRICE"] <= df["SALE PRICE"].quantile(0.99)
]["SALE PRICE"]

plt.figure(figsize=(11, 7))

sns.histplot(
    filtered_prices,
    bins=60,
    color="steelblue",
    edgecolor="white",
    stat="density",
    label="Histogram"
)

sns.kdeplot(
    filtered_prices,
    color=PURPLE,
    linewidth=2.5,
    label="KDE Density Curve"
)

plt.title(
    "Distribution of Sale Prices\n(up to the 99th percentile)",
    fontsize=18,
    fontweight="bold"
)
plt.xlabel("Sale Price ($)", fontsize=13)
plt.ylabel("Density", fontsize=13)
plt.legend()
plt.grid(True, alpha=0.4)
save_plot("price_distribution.png")


print("All graphs saved in results/graphs/")