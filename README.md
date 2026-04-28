
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

[1] https://www.kaggle.com/datasets/new-york-city/nyc-property-sales/data