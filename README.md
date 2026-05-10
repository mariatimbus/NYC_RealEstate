# NYC Real Estate Analysis — SVD, K-Means & Anomaly Detection

> Analiză completă a pieței imobiliare din New York City folosind descompunerea SVD (implementată manual), clustering K-Means, Ridge Regression și detectare de anomalii prin 3 metode independente (SVD reconstruction error, K-Means centroid distance, Isolation Forest).

---

## Dataset

**Sursă:** [NYC Property Sales — Kaggle](https://www.kaggle.com/datasets/new-york-city/nyc-property-sales/data)

Dataset-ul original conține ~84.000 de înregistrări de vânzări imobiliare din NYC. După curățare rămân **29.275 de proprietăți** cu 10 coloane relevante.

**Coloane folosite:**
| Coloană | Descriere |
|---------|-----------|
| `SALE PRICE` | Prețul de vânzare ($) — target |
| `NEIGHBORHOOD` | Cartierul din NYC |
| `BUILDING CLASS CATEGORY` | Categoria clădirii (condo, coop, etc.) |
| `BUILDING CLASS AT PRESENT` | Clasa actuală a clădirii |
| `TOTAL UNITS` | Număr total de unități |
| `RESIDENTIAL UNITS` | Unități rezidențiale |
| `COMMERCIAL UNITS` | Unități comerciale |
| `GROSS SQUARE FEET` | Suprafața construită brută |
| `LAND SQUARE FEET` | Suprafața terenului |
| `YEAR BUILT` | Anul construcției |

---

## Structura proiectului

```
NYC_RealEstate/
├── data/
│   ├── full_dataset.csv          # Dataset raw de la Kaggle (~84K rows)
│   ├── simplified_dataset.csv    # 10 coloane relevante
│   └── cleaned_dataset.csv       # Date curățate (29,275 rows × 10 cols)
├── results/
│   ├── descriptive_statistics.csv
│   ├── correlation_matrix.csv
│   ├── neighborhood_stats.csv
│   ├── building_class_stats.csv
│   ├── price_per_sqft_stats.csv
│   ├── kmeans_inertia.csv
│   ├── cluster_profiles.csv
│   ├── clustered_dataset.csv
│   ├── matrix_A.csv              # Feature matrix (29,275 × 408)
│   ├── vector_y.csv              # Target vector (29,275)
│   ├── anomaly_detection_comparison.csv
│   ├── undervalued_overvalued.csv
│   ├── svd_variance_explained.csv
│   ├── svd_loadings.csv
│   └── real_vs_predicted_svd.png
├── charts/
│   ├── 01_sale_price_distribution.png
│   ├── 02_boxplots_numeric.png
│   ├── 03_year_built_distribution.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_top_neighborhoods.png
│   ├── 06_top_building_classes.png
│   ├── 07_price_per_sqft.png
│   ├── 08_elbow_curve.png
│   ├── 09_pca_clusters.png
│   ├── 10_cluster_profiles.png
│   ├── 11_cluster_sizes.png
│   ├── 12_svd_scree_plot.png
│   ├── 13_svd_loadings_heatmap.png
│   ├── 14_svd_biplot.png
│   ├── 15_svd_group_contributions.png
│   ├── 16_anomaly_detection_comparison.png
│   ├── 17_undervalued_overvalued.png
│   ├── 18_ridge_feature_importance.png
│   ├── 19_anomaly_rate_by_cluster.png
│   ├── 20_under_over_anomaly_overlay.png
│   ├── 21_svd_error_vs_price.png
│   ├── 22_price_vs_year_clusters.png
│   └── 23_price_boxplot_by_cluster.png
├── src/
│   ├── load_kaggle.py            # Descarcă dataset de la Kaggle
│   ├── clean_data.py             # Curățare date
│   ├── exploratory_analysis.py   # Statistici + K-Means clustering
│   ├── build_matrix.py           # One-hot encoding + standardizare
│   ├── custom_svd.py             # Implementare manuală SVD
│   ├── svd_model.py              # Model de predicție cu SVD
│   ├── anomaly_detection.py      # Detectare anomalii (3 metode)
│   ├── undervalued_overvalued.py # Subevaluate / supraevaluate
│   └── generate_charts.py        # Generează toate vizualizările
├── personal/
│   └── find_perfect_properties.py # Căutare proprietăți după criterii personale
├── .gitignore
└── README.md                     # Acest fișier
```

---

## Pipeline

### Etapa 0 — Încărcare date (`src/load_kaggle.py`)

Descarcă dataset-ul complet de la Kaggle și salvează:
- `data/full_dataset.csv` — toate coloanele (~84K rows)
- `data/simplified_dataset.csv` — doar cele 10 coloane relevante

### Etapa 1 — Curățare date (`src/clean_data.py`)

Curăță `simplified_dataset.csv` și produce `cleaned_dataset.csv`:
- Elimină valori lipsă (`dropna`)
- Convertește coloanele numerice (coerce erori)
- Filtrează: `SALE PRICE > 0`, `TOTAL UNITS > 0`, `GROSS/LAND SQFT > 0`, `YEAR BUILT > 1800`
- Curăță coloanele categorice (elimină `""` și `"-"`)

**Rezultat:** 29.275 de proprietăți valide.

### Etapa 2 — Analiză statistică preliminară & Clustering (`src/exploratory_analysis.py`)

Generează statistici descriptive și aplică K-Means clustering:

| Rezultat | Fișier |
|----------|--------|
| Statistici descriptive | `results/descriptive_statistics.csv` |
| Matrice de corelație | `results/correlation_matrix.csv` |
| Statistici pe cartiere | `results/neighborhood_stats.csv` |
| Statistici pe tip clădire | `results/building_class_stats.csv` |
| Preț pe mp² | `results/price_per_sqft_stats.csv` |
| Inerție K-Means (elbow) | `results/kmeans_inertia.csv` |
| Profiluri clustere | `results/cluster_profiles.csv` |
| Dataset cu clustere | `results/clustered_dataset.csv` |

**Parametri clustering:**
- K = 4 (ales prin metoda cotului)
- Features: `SALE_PRICE_LOG`, `TOTAL UNITS`, `RESIDENTIAL UNITS`, `COMMERCIAL UNITS`, `GROSS_SQFT_LOG`, `LAND_SQFT_LOG`, `YEAR BUILT`
- Preprocesare: clip la percentila 1/99, log-transform, StandardScaler

**Clustere identificate:**
| Cluster | Proprietăți | Preț mediu | Descriere |
|---------|-------------|------------|-----------|
| **0** | ~26.400 | ~$821K | Proprietăți obișnuite, majoritatea dataset-ului |
| **1** | ~568 | ~$14.2M | Proprietăți de lux / comerciale mari |
| **2** | ~1.500 | ~$12.4M | Segment high-end |
| **3** | ~828 | ~$10–$3.700 | **Vânzări anomale** — transferuri intrafamiliale sau erori |

**Cea mai puternică corelație cu prețul:** `GROSS SQUARE FEET` (0.51).

### Etapa 3 — Construirea matricii (`src/build_matrix.py`)

Pregătește datele pentru modelul SVD:
- One-hot encode pe `NEIGHBORHOOD`, `BUILDING CLASS CATEGORY`, `BUILDING CLASS AT PRESENT`
- StandardScaler pe toate feature-urile
- Salvează `results/matrix_A.csv` (29.275 × 408) și `results/vector_y.csv`

### Etapa 4 — SVD Manual + Model de predicție

#### `src/custom_svd.py` — Implementare manuală SVD

Implementare de la zero a descompunerii SVD **fără a folosi `np.linalg.svd`**:

1. **Tridiagonalizare Householder** — transformă `AᵀA` într-o matrice tridiagonală simetrică
2. **Iterație QR implicită** — cu rotații Givens, shift Wilkinson și deflație
3. **Sortare valori proprii** — descrescător → obținem valorile singulare
4. **Construire U, Σ, V** — economy SVD (U: m×n, Σ: n×n, V: n×n)

**Optimizări cheie:**
- Economy SVD (U de dimensiune `m×n`, nu `m×m`) — evită alocarea a 51GB memorie
- Vectorizare în tridiagonalizare Householder
- Deflație în QR — convergență în ~736 iterații în ~15–20 secunde

#### `src/svd_model.py` — Predicție preț folosind SVD

Folosește SVD manual pentru a rezolva **linear least squares** prin pseudo-inversă:

```
x = A⁺ y = V · Σ⁻¹ · Uᵀ · y
y_pred = A · x
```

**Metrici:**
- **MAE:** ~$2.38M
- **RMSE:** ~$10.47M

Generează grafic dual:
- **Panou stânga:** vizualizare zoomată (< $50M)
- **Panou dreapta:** vizualizare log-log (toate valorile pozitive)

Salvează: `results/real_vs_predicted_svd.png`

### Etapa 5 — Detectare anomalii (`src/anomaly_detection.py`)

Compară **3 metode independente** de detectare a anomaliilor:

#### 1. SVD Reconstruction Error
- Rulează SVD (numpy) cu 5 componente
- Calculează eroarea de reconstrucție per proprietate
- Top 5% = anomalie

#### 2. K-Means Centroid Distance
- K-Means cu k=4
- Distanța euclidiană până la centroidul propriului cluster
- Top 5% = anomalie

#### 3. Isolation Forest (AI)
- `sklearn.ensemble.IsolationForest`
- `contamination=0.05`, `n_estimators=200`

**Rezultatele comparării:**
| Overlap | Proprietăți | Procent |
|---------|-------------|---------|
| Normal (toate 3 agrează) | ~27.800 | 95.0% |
| **Toate 3 metodele (anomalie)** | **538** | **1.84%** |
| SVD + K-Means (nu IF) | ~200 | — |
| SVD + IF (nu KM) | ~150 | — |
| K-Means + IF (nu SVD) | ~1.200 | — |

**Cel mai mare overlap:** K-Means ↔ Isolation Forest (**66.7%**).

Salvează: `results/anomaly_detection_comparison.csv` și `charts/16_anomaly_detection_comparison.png`

### Etapa 6 — Proprietăți subevaluate / supraevaluate (`src/undervalued_overvalued.py`)

#### Ridge Regression pe `log(SALE PRICE)`

Model: `log(price) ~ TOTAL_UNITS + RESIDENTIAL_UNITS + COMMERCIAL_UNITS + log(GROSS_SQFT) + log(LAND_SQFT) + YEAR_BUILT`

| Scope | R² (log) | Observații |
|-------|----------|------------|
| Toate datele | 0.063 | Slab — outlieri extreme ($1 → $2.2B) |
| Filtrat ($100K–$200M) | **0.49** | Mult mai fiabil |

#### Analiză per cluster (preț relativ)

Compară `PRICE_PER_SQFT` al fiecărei proprietăți cu mediana cluster-ului propriu:
- **Undervalued:** `$/sqft` mult sub mediana clusterului
- **Overvalued:** `$/sqft` mult peste mediana clusterului

#### Conectare cu anomaliile

Anomaliile detectate de SVD, K-Means și Isolation Forest acoperă **90–100%** din proprietățile extreme subevaluate/supraevaluate — confirmând consistența metodelor.

Salvează: `results/undervalued_overvalued.csv` și `charts/17_undervalued_overvalued.png`

### Etapa 7 — Căutare proprietăți personale (`personal/find_perfect_properties.py`)

Script personal pentru căutarea de proprietăți în Manhattan după criterii custom:

| Criteriu | Valoare |
|----------|---------|
| Borough | Manhattan (1) |
| Suprafață | ≥ 500 sqft |
| Preț | $500K – $10M |
| Unități | ≥ 4 (proxy pentru dormitoare) |
| Cartiere | 28 de cartiere selectate (Upper East/West, Tribeca, SoHo, etc.) |

**Rezultat:** 128 de proprietăți matching, sortate după `price/sqft` crescător.

> **Notă:** Dataset-ul NYC Rolling Sales este la nivel de **clădire**, nu de apartament individual. Nu există coloane pentru bedroom/bathroom/HOA.

### Etapa 8 — Vizualizări suplimentare

Pe lângă graficele de bază (01–17), au fost generate 6 vizualizări avansate:

| # | Grafic | Insight |
|---|--------|---------|
| **18** | Feature Importance (Ridge) | `GROSS_SQFT_LOG` domină; `RESIDENTIAL_UNITS` și `YEAR_BUILT` au coeficienți negativi |
| **19** | Anomaly Rate per Cluster | Cluster 2 = 91.5% anomalii SVD, 100% IF; Clusterele 0 și 3 sunt "curate" |
| **20** | Underpriced vs Overpriced vs Anomalies | 466 undervalued, 1.452 overvalued, 1.192 anomalii confirmate |
| **21** | SVD Error vs Price | Eroare de reconstrucție crescută indiferent de preț — outlieri structurali |
| **22** | Price vs Year Built + Clusters | Cluster 3 = proprietăți noi; Cluster 0 = proprietăți vechi cu volatilitate |
| **23** | Price Boxplot by Cluster | Mediana cea mai înaltă în Cluster 2 (~$8M); Cluster 3 cel mai compact |

---

## Cum să rulezi

### 1. Setup mediu

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib seaborn kagglehub
```

### 2. Pipeline complet

```bash
# Etapa 0 — Încărcare date
python src/load_kaggle.py

# Etapa 1 — Curățare
python src/clean_data.py

# Etapa 2 — Analiză statistică + clustering
python src/exploratory_analysis.py

# Etapa 3 — Construire matrice
python src/build_matrix.py

# Etapa 4 — SVD manual + predicție
python src/svd_model.py

# Etapa 5 — Detectare anomalii
python src/anomaly_detection.py

# Etapa 6 — Subevaluate / supraevaluate
python src/undervalued_overvalued.py

# Etapa 7 — Proprietăți personale (opțional)
python personal/find_perfect_properties.py

# Etapa 8 — Generare grafice
python src/generate_charts.py
```

---

## Dependințe

| Pachet | Versiune minimă | Utilizare |
|--------|-----------------|-----------|
| `numpy` | — | SVD manual, algebra liniară |
| `pandas` | — | Manipulare date |
| `scikit-learn` | — | K-Means, Isolation Forest, Ridge, StandardScaler |
| `matplotlib` | — | Vizualizări |
| `seaborn` | — | Heatmap, histograme, boxplot |
| `kagglehub` | — | Descărcare dataset |

---

## Limitări & Note

1. **Dataset la nivel de clădire** — nu există coloane pentru bedroom, bathroom, etaj, sau HOA fees. `TOTAL_UNITS >= 4` este folosit ca proxy pentru mărime.

2. **Outlieri extreme** — prețuri de $1 (transferuri nominale) până la $2.2B afectează semnificativ regresia pe datele raw. Filtrarea ($100K–$200M) îmbunătățește R² de la 0.06 la 0.49.

3. **SVD manual** — implementarea funcționează corect dar este semnificativ mai lentă decât LAPACK (`np.linalg.svd`). Scopul principal este didactic: înțelegerea algoritmilor din spatele SVD.

4. **Economy SVD** — U are dimensiunea `(m, n)` nu `(m, m)`. Această alegere reduce memoria de la ~51GB la ~100MB.

5. **QR implicit cu Givens** — a înlocuit QR explicit (care nu convergea pe 408×408) cu rotații Givens + deflație, convergând în ~736 iterații.

---

