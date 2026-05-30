# Fundații Matematice ale Descompunerii Valorilor Singulare (SVD)

> Document detaliat despre implementarea manuală a SVD din proiectul NYC Real Estate. Acoperă întregul pipeline matematic, de la tridiagonalizare Householder până la iterația QR implicită cu rotații Givens și predicția prețurilor prin pseudo-inversă.

---

## 1. Definiția SVD

Pentru orice matrice **A** ∈ ℝ^(m×n) cu m ≥ n, există descompunerea:

```
A = U · Σ · Vᵀ
```

unde:
- **U** ∈ ℝ^(m×m) — matrice ortogonală (vectori singulari la stânga)
- **Σ** ∈ ℝ^(m×n) — matrice diagonală cu valorile singulare σ₁ ≥ σ₂ ≥ ... ≥ σₙ ≥ 0 pe diagonală
- **V** ∈ ℝ^(n×n) — matrice ortogonală (vectori singulari la dreapta)

### Interpretare în contextul proiectului

| Componentă | Interpretare |
|------------|--------------|
| **U** | Preferințele pieței / profilele cumpărătorilor (29.275 proprietăți × componente) |
| **Σ** | Importanța fiecărei componente (scalară) |
| **Vᵀ** | Relațiile dintre caracteristicile proprietăților (408 feature-uri × componente) |

---

## 2. Pipeline-ul implementat

Implementarea noastră NU folosește `np.linalg.svd`. Algoritmul de la zero este:

```
A (29.275 × 408)
    │
    ▼
B = AᵀA  (simetrică, 408 × 408)
    │
    ▼
Tridiagonalizare Householder  →  Q₀, T₀
    │
    ▼
Iterație QR implicită  →  D (valori proprii), V (vectori proprii)
    │
    ▼
Valori singulare: σᵢ = √λᵢ  (sortate descrescător)
    │
    ▼
U = A·V·Σ⁻¹  (economy: m×n, nu m×m)
    │
    ▼
Pseudo-inversă: A⁺ = V·Σ⁻¹·Uᵀ
    │
    ▼
x = A⁺·y  (coeficienți de regresie)
y_pred = A·x  (prețuri estimate)
```

---

## 3. Pasul 1: Construirea lui B = AᵀA

### De ce AᵀA?

Valorile singulare ale lui **A** sunt rădăcinile pătrate ale valorilor proprii ale lui **AᵀA**:

```
A = UΣVᵀ
AᵀA = (UΣVᵀ)ᵀ(UΣVᵀ) = VΣᵀUᵀUΣVᵀ = VΣ²Vᵀ
```

Deci **AᵀA = VΣ²Vᵀ**, care este o descompunere spectrală:
- Coloanele lui **V** sunt vectorii proprii ai lui AᵀA
- Valorile proprii λᵢ = σᵢ² sunt pătratele valorilor singulare

### Cod:

```python
def SVD(A, TOL=1e-14):
    A = np.copy(A).astype(float)
    m, n = np.shape(A)
    
    # Pasul 1: B = AᵀA
    B = A.T @ A           # B ∈ ℝ^(n×n), simetrică, pozitiv semi-definită
```

**Dimensiuni în proiect:** B = 408 × 408 (nu 29.275 × 29.275, ceea ce ar fi enorm).

---

## 4. Pasul 2: Tridiagonalizare Householder

### 4.1 Problema

B este simetrică dar densă. Vrem să o aducem la formă **tridiagonală simetrică**:

```
    ⎡ d₁  e₁   0    0   ...  0 ⎤
    ⎢ e₁  d₂  e₂    0   ...  0 ⎥
T = ⎢  0  e₂  d₃   e₃   ...  0 ⎥
    ⎢  :   :   :    :   ⋱   : ⎥
    ⎣  0   0   0   eₙ₋₁  dₙ ⎦
```

Aceasta permite ca pasul QR să ruleze în O(n²) per iterație în loc de O(n³).

### 4.2 Reflectorul Householder

Pentru un vector **x** ∈ ℝⁿ, reflectorul Householder este:

```
v = x + sign(x₁) · ||x||₂ · e₁
H = I - 2 · (vvᵀ)/(vᵀv)
```

Proprietăți:
- **H** este ortogonală: HᵀH = I
- **H·x** are doar prima componentă nenulă (reflectă x pe e₁)
- Aplicarea H la o matrice se face fără a construi H explicit (O(n²), nu O(n³))

### 4.3 Algoritmul de tridiagonalizare

Pentru k = 1, ..., n-2:
1. Extrage coloana k sub diagonală: x = T[k+1:n, k]
2. Construiește reflectorul H_k care anulează toate elementele lui x exceptând primul
3. Aplică H_k simultan din stânga și dreapta: T = H_k · T · H_k
4. Acumulează transformarea: Q = Q · H_k

### 4.4 Cod detaliat

```python
def Tridiag_Householder(A):
    n = A.shape[0]
    T = np.copy(A).astype(float)
    Q = np.eye(n)

    for k in range(n - 2):
        # Extragem coloana sub-diagonală
        x = T[k + 1:, k].copy()
        norm_x = np.linalg.norm(x)

        if abs(norm_x) < 1e-14:
            continue

        # Construim vectorul Householder
        sgn = np.sign(x[0]) if abs(x[0]) > 1e-14 else 1.0
        x[0] += sgn * norm_x
        v = x / np.linalg.norm(x)

        # Aplicăm reflectorul din stânga: T = H @ T
        # H = I - 2·v·vᵀ  ⇒  H·T = T - 2·v·(vᵀ·T)
        T[k + 1:, :] -= 2 * np.outer(v, v @ T[k + 1:, :])

        # Aplicăm reflectorul din dreapta: T = T @ H
        # T·H = T - 2·(T·v)·vᵀ
        T[:, k + 1:] -= 2 * np.outer(T[:, k + 1:] @ v, v)

        # Acumulăm transformările ortogonale în Q
        Q[:, k + 1:] -= 2 * np.outer(Q[:, k + 1:] @ v, v)

    # Forțăm simetrie exactă (curățăm zgomot numeric)
    T = np.tril(np.triu(T, -1), 1)
    T = (T + T.T) / 2

    return Q, T
```

### 4.5 De ce funcționează

Fiecare iterație k anulează elementele de sub diagonala k fără a reintroduce zerourile deja create. După n-2 pași, matricea are forma tridiagonală dorită.

**Complexitate:** O(n³) dar cu constantă mică (~2/3 n³ operații), vectorizată în NumPy.

---

## 5. Pasul 3: Iterație QR Implicită

### 5.1 Problema

Avem matricea tridiagonală T. Vrem să-i găsim valorile proprii (care sunt valorile proprii ale lui B = AᵀA).

Metoda QR clasică:
```
T₀ = T
Tₖ₊₁ = RₖQₖ  unde Tₖ = QₖRₖ (descompunere QR)
```

Converge foarte lent. Soluția: **QR implicit cu shift Wilkinson**.

### 5.2 Shift Wilkinson

În loc să aplicăm QR pe T, aplicăm pe **T - μI**, unde μ este o estimare a valorii proprii:

```
μ = dₙ - eₙ₋₁² / (δ + sign(δ)·√(δ² + eₙ₋₁²))
unde δ = (dₙ₋₁ - dₙ)/2
```

Acest shift accelerează convergența cu un ordin de mărime.

### 5.3 Rotații Givens

Pentru o matrice tridiagonală, QR se face eficient cu **rotații Givens** (nu Gram-Schmidt).

O rotație Givens în planul (i, j) anulează elementul j:

```
G(i,j,θ) =  ⎡ 1           ⎤
            ⎢    ⋱        ⎥
            ⎢      c   s  ⎥  ← rând i
            ⎢     -s   c  ⎥  ← rând j
            ⎢        ⋱    ⎥
            ⎣           1 ⎦

c = a / √(a² + b²),   s = b / √(a² + b²)
```

Pentru matrice tridiagonală:
- Parcurgem sub-diagonala de jos în sus
- Fiecare rotație anulează un element sub-diagonal
- Rezultatul R este superior triunghiular

### 5.4 QR Implicit (fără a forma explicit T - μI)

Algoritmul **implicit** aplică prima rotație Givens direct pe T, fără a calcula explicit T - μI. Aceasta evită probleme numerice.

### 5.5 Deflație

Când un element sub-diagonal |eₖ| < TOL, considerăm că valorile proprii din blocul de jos au convergit. "Tăiem" blocul și continuăm doar pe partea de sus.

```
    ⎡ T₁₁   0 ⎤
T = ⎢         ⎥
    ⎣  0   T₂₂⎦
         ↑
    eₖ ≈ 0  ⇒  deflație: rezolvăm separat T₁₁ și T₂₂
```

### 5.6 Cod detaliat

```python
def _qr_step_tridiag_explicit(T_block):
    """Un pas QR cu shift Wilkinson pe un bloc tridiagonal."""
    n = T_block.shape[0]
    T_shifted = T_block.copy()

    # Shift Wilkinson
    delta = (T_shifted[n - 2, n - 2] - T_shifted[n - 1, n - 1]) / 2.0
    sign = np.sign(delta) if abs(delta) > 1e-14 else 1.0
    mu = T_shifted[n - 1, n - 1] - T_shifted[n - 2, n - 1] ** 2 / (
        delta + sign * np.sqrt(delta ** 2 + T_shifted[n - 2, n - 1] ** 2)
    )
    T_shifted -= mu * np.eye(n)

    rotations = []

    # Rotații din stânga: construim R = Qᵀ(T - μI)
    for k in range(n - 1):
        a = T_shifted[k, k]
        b = T_shifted[k + 1, k]
        r = np.hypot(a, b)
        if r < 1e-14:
            rotations.append((1.0, 0.0))
            continue
        c = a / r
        s = b / r

        row_k = T_shifted[k, :].copy()
        row_k1 = T_shifted[k + 1, :].copy()
        T_shifted[k, :] = c * row_k + s * row_k1
        T_shifted[k + 1, :] = -s * row_k + c * row_k1
        rotations.append((c, s))

    # Rotații din dreapta: T' = R·Q + μI
    for k in range(n - 1):
        c, s = rotations[k]
        col_k = T_shifted[:, k].copy()
        col_k1 = T_shifted[:, k + 1].copy()
        T_shifted[:, k] = c * col_k + s * col_k1
        T_shifted[:, k + 1] = -s * col_k + c * col_k1

    T_new = T_shifted + mu * np.eye(n)
    
    # Curățăm zgomotul numeric sub banda tridiagonală
    T_new = np.tril(np.triu(T_new, -1), 1)
    T_new = (T_new + T_new.T) / 2

    return T_new, rotations
```

```python
def QR_iteration(A, Q, TOL=1e-6):
    """Eigendecompunere prin QR implicit cu deflație."""
    T = Q.T @ A @ Q      # Aducem A la forma tridiagonală
    n = A.shape[0]
    V = Q.copy()         # Acumulator pentru vectorii proprii

    active_end = n
    total_iters = 0
    max_total_iters = 5000

    while active_end > 1 and total_iters < max_total_iters:
        # DEFLAȚIE: verificăm dacă elementul de sub diagonală a convergit
        while active_end > 1 and abs(T[active_end - 1, active_end - 2]) < TOL:
            active_end -= 1

        if active_end <= 1:
            break

        # Aplicăm pași QR pe blocul activ
        block_iters = 0
        while abs(T[active_end - 1, active_end - 2]) >= TOL and block_iters < 10:
            T_block = T[:active_end, :active_end].copy()
            T_block_new, rots = _qr_step_tridiag_explicit(T_block)
            T[:active_end, :active_end] = T_block_new

            # Acumulăm rotațiile în matricea de vectori proprii
            for k, (c, s) in enumerate(rots):
                v_k = V[:, k].copy()
                v_k1 = V[:, k + 1].copy()
                V[:, k] = c * v_k + s * v_k1
                V[:, k + 1] = -s * v_k + c * v_k1

            total_iters += 1
            block_iters += 1

        # Forțăm zero pe elementul deflat
        if active_end > 1:
            T[active_end - 1, active_end - 2] = 0.0
            T[active_end - 2, active_end - 1] = 0.0
            active_end -= 1

    D = np.diag(np.diag(T))
    return D, V
```

### 5.7 De ce "explicit" în nume funcției

Funcția `_qr_step_tridiag_explicit` calculează efectiv `T - μI` explicit (scade μ de pe diagonală). Algoritmul QR **complet implicit** evită și acest pas, folosind doar prima rotație Givens calculată din elementul (2,1) al matricei `(T - μI)`. Implementarea noastră este "semi-implicită" — shift-ul e aplicat explicit, dar QR-ul pe bloc tridiagonal e eficient.

**Performanță:** ~736 iterații pentru matricea 408×408, ~15–20 secunde.

---

## 6. Pasul 4: Valori Singulare și V

După QR iteration avem:
- **D** = diag(λ₁, ..., λₙ) — valorile proprii ale lui AᵀA
- **V** = matricea de vectori proprii

Valorile singulare sunt:

```
σᵢ = √λᵢ,  pentru i = 1, ..., n
```

Sortăm descrescător:

```python
val = np.diag(D)
indici = np.argsort(val)[::-1]
val = val[indici]
V = V[:, indici]

# Protecție împotriva erorilor numerice (valori proprii ușor negative)
val[val < 0] = 0
sigma = np.sqrt(val)

S = np.zeros((n, n))
for i in range(n):
    S[i, i] = sigma[i]
```

---

## 7. Pasul 5: Calculul lui U (Economy SVD)

### 7.1 Formula teoretică

Din A = UΣVᵀ, înmulțind cu V la dreapta:

```
AV = UΣ  ⇒  U = AVΣ⁻¹
```

Deci coloana i a lui U:

```
uᵢ = (A · vᵢ) / σᵢ,   pentru σᵢ > 0
```

### 7.2 Economy SVD

**Problema:** U "full" are dimensiunea m×m = 29.275 × 29.275 ≈ **856 milioane de elemente** (~6.8 GB în float64).

**Soluția:** Folosim **economy SVD**:
- U: m×n = 29.275 × 408 (~12 milioane elemente, ~96 MB)
- Σ: n×n = 408 × 408
- V: n×n = 408 × 408

Aceasta este suficientă pentru majoritatea aplicațiilor (inclusiv least squares).

### 7.3 Cod

```python
U = np.zeros((m, n))
for i in range(n):
    if sigma[i] > TOL:
        U[:, i] = (A @ V[:, i]) / sigma[i]
```

---

## 8. Pasul 6: Least Squares prin Pseudo-inversă

### 8.1 Problema

Vrem să rezolvăm:

```
min ||Ax - y||₂²
```

unde A ∈ ℝ^(m×n), y ∈ ℝ^m, x ∈ ℝ^n.

### 8.2 Soluția prin SVD

Soluția în sensul celor mai mici pătrate este:

```
x = A⁺y = VΣ⁺Uᵀy
```

unde Σ⁺ este pseudo-inversa lui Σ:

```
Σ⁺ᵢᵢ = 1/σᵢ  dacă σᵢ > ε
       0     altfel
```

### 8.3 Cod

```python
# Construim Σ⁺ (pseudo-inversa)
sigma_inv = np.zeros((S.shape[1], S.shape[0]))  # n×n
for i in range(min(S.shape)):
    if S[i, i] > 1e-10:
        sigma_inv[i, i] = 1 / S[i, i]

# Pseudo-inversa: A⁺ = V · Σ⁺ · Uᵀ
A_pinv = V @ sigma_inv @ U.T

# Coeficienții de regresie
x = A_pinv @ y

# Predicții
y_pred = A @ x
```

### 8.4 Metrici de evaluare

```
MAE  = (1/m) · Σ|yᵢ - ŷᵢ|
RMSE = √((1/m) · Σ(yᵢ - ŷᵢ)²)
```

Rezultate pe dataset:
- **MAE:** ~$2.38M
- **RMSE:** ~$10.47M

Erorile mari sunt datorate outlier-ilor extreme (prețuri de $1 până la $2.2B).

---

## 9. Analiza Complexității

| Pas | Operații dominante | Complexitate |
|-----|-------------------|--------------|
| B = AᵀA | înmulțire matricială | O(m·n²) = O(29K · 408²) |
| Tridiagonalizare Householder | n-2 reflectori, fiecare O(n²) | O(n³) = O(408³) |
| QR iteration | ~736 iterații × O(n²) per iterație | O(k·n²) ≈ O(736 · 408²) |
| Sortare valori proprii | quicksort | O(n log n) |
| Calcul U | m·n înmulțiri vector-matrice | O(m·n²) |
| Pseudo-inversă + predicție | înmulțiri matriciale | O(m·n²) |

**Complexitate totală:** dominată de O(m·n²) pentru AᵀA și calculul lui U.

**Memorie:**
- Full SVD: U(m×m) = 29.275² × 8 bytes ≈ **6.8 GB** ❌
- Economy SVD: U(m×n) = 29.275 × 408 × 8 bytes ≈ **96 MB** ✅

---

## 10. Comparatie cu np.linalg.svd

| Aspect | Implementare manuală | np.linalg.svd (LAPACK) |
|--------|---------------------|------------------------|
| Algoritm | Householder + QR implicit | Bidiagonalizare Golub-Kahan + QR divide-and-conquer |
| Timp (408×408) | ~15–20 secunde | < 0.1 secunde |
| Precizie | Bună | Excelentă (IEEE double) |
| Scop | Didactic / înțelegere algoritm | Producție |
| Memorie U | Economy (m×n) | Full sau economy |

---

## 11. Detectare Anomalii cu SVD

Folosim SVD (numpy, rapid) pentru detectarea anomaliilor prin **reconstruction error**:

```python
U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)

# Păstrăm doar k componente principale
U_k = U[:, :k]
s_k = s[:k]
Vt_k = Vt[:k, :]

X_reconstructed = U_k @ np.diag(s_k) @ Vt_k

# Eroarea de reconstrucție
error = ||X_scaled - X_reconstructed||₂² per rând

# Top 5% = anomalii
threshold = percentile(error, 95)
anomalies = error > threshold
```

Intuiție: proprietățile "normale" se proiectează bine în primele k componente. Anomaliile au o structură diferită și nu se reconstruiesc bine.

---

## 12. Rezumat Matematic

```
┌─────────────────────────────────────────────────────────────┐
│  A ∈ ℝ^(m×n)   (29.275 × 408)                               │
│                                                             │
│  B = AᵀA  →  simetrică, pozitiv semi-definită               │
│                                                             │
│  Householder:  B = Q₀T₀Q₀ᵀ   (T₀ tridiagonală)              │
│                                                             │
│  QR iteration:  Tₖ → D (diagonală)                         │
│                 V = Q₀ · (produs rotații Givens)            │
│                                                             │
│  σᵢ = √λᵢ   (valori singulare)                              │
│  uᵢ = Avᵢ/σᵢ   (vectori singulari stânga)                  │
│                                                             │
│  A = UΣVᵀ   (descompunere SVD)                              │
│                                                             │
│  x = VΣ⁺Uᵀy   (soluție least squares)                       │
│  ŷ = Ax       (predicție preț)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Referințe

1. Golub, G.H. & Van Loan, C.F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
2. Trefethen, L.N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
3. Demmel, J.W. (1997). *Applied Numerical Linear Algebra*. SIAM.
