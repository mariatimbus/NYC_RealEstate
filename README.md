
Vom folosi descompunerea:
A = UΣVᵀ
unde interpretăm:
U — preferințele pieței / cumpărătorilor
Σ — importanța componentelor
Vᵀ — relațiile dintre caracteristicile proprietăților
Caracteristicile alese ar fi:

U:
- SALE PRICE: 
- NEIGHBORHOOD
- BUILDING CLASS CATEGORY

Σ:
- TOTAL UNITS
- RESIDENTIAL UNITS
- COMMERCIAL UNITS

Vᵀ:
- GROSS SQUARE FEET
- LAND SQUARE FEET
- YEAR BUILT
- BUILDING CLASS AT PRESENT

Proiectul ar include:
1. Data cleaning — pregătirea datelor pentru analiză (am observat in preprocessing ca unele date sunt lipsa, asa ca vom curățarea acele valori prin eliminarea intregului row; avem la dispozitie 67k date)
2. Analiză statistică preliminară și clustering — agregarea datelor pe principalele axe descriptive ale datasetului și calcularea valorilor medii pentru a identifica dependențele dintre caracteristicile proprietăților și prețul de vânzare. Ulterior, dorim să aplicăm un algoritm de învățare nesupervizată, K-Means, pentru a grupa proprietățile în funcție de similarități.
3. Model SVD — construirea unui model de predicție a prețurilor folosind descompunerea SVD.
4. Evaluarea modelului — compararea prețului real cu prețul estimat.
5. Identificarea anomaliilor — evidențierea proprietăților subevaluate sau supraevaluate.

---

## Rulare

### Etapa 2 — Analiză statistică preliminară și clustering

Script: `src/exploratory_analysis.py`

```bash
cd /Users/maria/NYC_RealEstate
.venv/bin/python src/exploratory_analysis.py
```

Acest script preia `data/cleaned_dataset.csv` și generează următoarele rezultate în directorul `results/`:

1. **Statistici descriptive** (`descriptive_statistics.csv`) — count, mean, std, min, max, mediană, skewness pentru toate variabilele numerice.
2. **Analiza corelațiilor** (`correlation_matrix.csv`) — matricea de corelație; evidențiază dependențele dintre caracteristici și prețul de vânzare.
3. **Agregări pe cartiere** (`neighborhood_stats.csv`) — preț mediu și median pe fiecare cartier.
4. **Agregări pe tip de clădire** (`building_class_stats.csv`) — preț mediu și median pe fiecare categorie de clasă a clădirii.
5. **Preț pe mp²** (`price_per_sqft_stats.csv`) — statistici pentru prețul pe GROSS SQFT și LAND SQFT.
6. **Clustering K-Means** (`kmeans_inertia.csv`, `cluster_profiles.csv`, `clustered_dataset.csv`):
   - Aplică metoda cotului pentru a alege numărul optim de clustere (k=4).
   - Normalizează datele și folosește log-transform pentru variabilele skewed.
   - Grupează proprietățile în 4 segmente distincte.
   - Calculează varianța explicată de PCA.

### Insight-uri principale din Etapa 2

- **Cea mai puternică corelație cu prețul:** `GROSS SQUARE FEET` (0.51).
- **Top cartiere după preț mediu:** Midtown CBD, Financial, Fashion.
- **Clustere identificate:**
  - **Cluster 0** (~26.4K prop.) — proprietăți obișnuite, preț mediu ~$821K.
  - **Cluster 1** (~568 prop.) — proprietăți de lux/comerciale mari, preț mediu ~$14.2M.
  - **Cluster 2** (~1.5K prop.) — segment high-end, preț mediu ~$12.4M.
  - **Cluster 3** (~828 prop.) — **vânzări anomale** (prețuri ~$10–$3,700), posibil transferuri intrafamiliale sau erori de înregistrare — relevant pentru Etapa 5.

[1] https://www.kaggle.com/datasets/new-york-city/nyc-property-sales/data
