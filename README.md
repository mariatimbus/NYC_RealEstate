# NYC Real Estate Analysis

Machine learning and statistical analysis of the **NYC Property Sales** dataset using **SVD, K-Means, Ridge Regression, and anomaly detection**.

## Overview

This project analyzes NYC real estate transactions to:

* Explore property price patterns
* Segment properties with K-Means
* Detect unusual transactions
* Identify potentially undervalued and overvalued properties
* Implement Singular Value Decomposition (SVD) from scratch

## Dataset

**NYC Property Sales — Kaggle**

* ~84,000 raw transactions
* **29,275** properties after cleaning
* Features include price, neighborhood, building type, square footage, units, and year built

## Methods

### Custom SVD

SVD is implemented from scratch without `np.linalg.svd`, using:

* Householder transformations
* Implicit QR iteration
* Givens rotations
* Wilkinson shifts
* Deflation

Economy SVD is used to reduce memory usage.

### K-Means Clustering

Properties are grouped into **4 market segments** based on:

* Sale price
* Square footage
* Number of units
* Year built

### Anomaly Detection

Three independent methods are compared:

* SVD reconstruction error
* K-Means centroid distance
* Isolation Forest

**538 properties** were identified as anomalies by all three methods.

### Property Valuation

Ridge Regression and price-per-square-foot analysis are used to detect potentially:

* Undervalued properties
* Overvalued properties

## Results

* **29,275** cleaned properties
* Strongest price correlation: **Gross Square Feet — 0.51**
* **538** anomalies confirmed by all three detection methods
* K-Means / Isolation Forest overlap: **66.7%**
* Ridge Regression on filtered data: **R² = 0.49**
* Custom SVD converges in approximately **736 QR iterations**

## Project Structure

```text
NYC_RealEstate/
├── data/
├── results/
├── charts/
├── src/
│   ├── load_kaggle.py
│   ├── clean_data.py
│   ├── exploratory_analysis.py
│   ├── build_matrix.py
│   ├── custom_svd.py
│   ├── svd_model.py
│   ├── anomaly_detection.py
│   ├── undervalued_overvalued.py
│   └── generate_charts.py
└── README.md
```

## Installation

```bash
pip install numpy pandas scikit-learn matplotlib seaborn kagglehub
```

## Run

```bash
python src/load_kaggle.py
python src/clean_data.py
python src/exploratory_analysis.py
python src/build_matrix.py
python src/svd_model.py
python src/anomaly_detection.py
python src/undervalued_overvalued.py
python src/generate_charts.py
```

## Tech Stack

**Python · NumPy · Pandas · scikit-learn · Matplotlib · Seaborn**
