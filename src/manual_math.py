"""
manual_math.py — Implementări pure Python (fără numpy) pentru operații
matriceale și statistici folosite în proiectul NYC Real Estate.

Scop: înlocuirea cât mai multor comenzi np.* cu implementări "de mână".
"""

import math
import random


# ---------------------------------------------------------------------------
# 1. Operații de bază pe matrice (listă de liste)
# ---------------------------------------------------------------------------

def mat_shape(A):
    """Returnează (m, n) pentru o matrice A (listă de liste)."""
    if not A:
        return (0, 0)
    return (len(A), len(A[0]))


def mat_zeros(m, n):
    """Creează o matrice m×n cu zero-uri."""
    return [[0.0 for _ in range(n)] for _ in range(m)]


def mat_eye(n):
    """Creează matricea identitate I_n."""
    I = mat_zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I


def mat_copy(A):
    """Deep copy pentru matrice."""
    return [row[:] for row in A]


def mat_transpose(A):
    """Transpusa matricei A."""
    m, n = mat_shape(A)
    T = mat_zeros(n, m)
    for i in range(m):
        for j in range(n):
            T[j][i] = A[i][j]
    return T


def mat_mul(A, B):
    """Înmulțirea a două matrice: A @ B."""
    m, n = mat_shape(A)
    p, q = mat_shape(B)
    if n != p:
        raise ValueError(f"Dimensiuni incompatibile: {m}x{n} @ {p}x{q}")
    C = mat_zeros(m, q)
    for i in range(m):
        for k in range(n):
            aik = A[i][k]
            if aik != 0.0:
                row_b = B[k]
                row_c = C[i]
                for j in range(q):
                    row_c[j] += aik * row_b[j]
    return C


def mat_vec_mul(A, v):
    """Înmulțire matrice-vector: A @ v. v este listă."""
    m, n = mat_shape(A)
    if len(v) != n:
        raise ValueError("Dimensiuni incompatibile matrice-vector")
    result = [0.0] * m
    for i in range(m):
        s = 0.0
        row = A[i]
        for j in range(n):
            s += row[j] * v[j]
        result[i] = s
    return result


def vec_mat_mul(v, A):
    """Înmulțire vector-matrice (vector linie): v @ A."""
    m, n = mat_shape(A)
    if len(v) != m:
        raise ValueError("Dimensiuni incompatibile vector-matrice")
    result = [0.0] * n
    for j in range(n):
        s = 0.0
        for i in range(m):
            s += v[i] * A[i][j]
        result[j] = s
    return result


def mat_sub(A, B):
    """Scădere element-cu-element: A - B."""
    m, n = mat_shape(A)
    C = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] - B[i][j]
    return C


def mat_add(A, B):
    """Adunare element-cu-element: A + B."""
    m, n = mat_shape(A)
    C = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] + B[i][j]
    return C


def mat_scalar_mul(A, s):
    """Înmulțire cu scalar: s * A."""
    m, n = mat_shape(A)
    C = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] * s
    return C


def mat_diag(A):
    """Extrage diagonala principală a matricei A ca listă."""
    m, n = mat_shape(A)
    d = []
    for i in range(min(m, n)):
        d.append(A[i][i])
    return d


def mat_from_diag(d):
    """Creează matrice diagonală dintr-o listă."""
    n = len(d)
    A = mat_zeros(n, n)
    for i in range(n):
        A[i][i] = d[i]
    return A


def mat_tril(A, k=0):
    """Păstrează doar elementele de sub diagonala k (inclusiv). Restul 0."""
    m, n = mat_shape(A)
    R = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            if j <= i + k:
                R[i][j] = A[i][j]
    return R


def mat_triu(A, k=0):
    """Păstrează doar elementele deasupra diagonalei k (inclusiv). Restul 0."""
    m, n = mat_shape(A)
    R = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            if j >= i + k:
                R[i][j] = A[i][j]
    return R


def mat_get_col(A, j):
    """Returnează coloana j ca listă."""
    return [row[j] for row in A]


def mat_set_col(A, j, col):
    """Setează coloana j dintr-o listă. Modifică A in-place."""
    for i in range(len(A)):
        A[i][j] = col[i]


def mat_get_row(A, i):
    """Returnează linia i ca listă."""
    return A[i][:]


def mat_set_row(A, i, row):
    """Setează linia i dintr-o listă. Modifică A in-place."""
    A[i] = row[:]


# ---------------------------------------------------------------------------
# 2. Operații pe vectori
# ---------------------------------------------------------------------------

def vec_dot(u, v):
    """Produs scalar între doi vectori."""
    return sum(ui * vi for ui, vi in zip(u, v))


def vec_norm(v):
    """Norma Euclidiană (L2) a unui vector."""
    return math.sqrt(sum(xi * xi for xi in v))


def vec_sub(u, v):
    """Scădere vectorială: u - v."""
    return [ui - vi for ui, vi in zip(u, v)]


def vec_add(u, v):
    """Adunare vectorială: u + v."""
    return [ui + vi for ui, vi in zip(u, v)]


def vec_scalar_mul(v, s):
    """Înmulțire vector cu scalar."""
    return [xi * s for xi in v]


def vec_outer(u, v):
    """Produs exterior: u * v^T. Returnează matrice."""
    m = len(u)
    n = len(v)
    A = mat_zeros(m, n)
    for i in range(m):
        ui = u[i]
        row = A[i]
        for j in range(n):
            row[j] = ui * v[j]
    return A


def vec_sign(x):
    """Semnul unui scalar."""
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    return 0.0


def vec_sign_v(v):
    """Semn element-cu-element pentru un vector."""
    return [vec_sign(x) for x in v]


def vec_sqrt(v):
    """Rădăcină pătrată element-cu-element."""
    return [math.sqrt(x) if x > 0 else 0.0 for x in v]


def vec_abs(v):
    """Modul element-cu-element."""
    return [abs(x) for x in v]


def vec_sum(v):
    """Suma elementelor unui vector."""
    return sum(v)


def vec_maximum(a, b):
    """Element-wise max (suportă scalar sau listă)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return max(a, b)
    return [max(ai, bi) for ai, bi in zip(a, b)]


def vec_hypot(a, b):
    """sqrt(a^2 + b^2) — evită overflow."""
    return math.hypot(a, b)


# ---------------------------------------------------------------------------
# 3. Statistici și utilitare (pure Python)
# ---------------------------------------------------------------------------

def mean(arr):
    """Media aritmetică."""
    if not arr:
        return 0.0
    return sum(arr) / len(arr)


def std_dev(arr, ddof=0):
    """Deviația standard."""
    if len(arr) <= ddof:
        return 0.0
    m = mean(arr)
    variance = sum((x - m) ** 2 for x in arr) / (len(arr) - ddof)
    return math.sqrt(variance)


def percentile(arr, p):
    """Percentila p (0-100) folosind interpolare liniară."""
    if not arr:
        return 0.0
    sorted_arr = sorted(arr)
    n = len(sorted_arr)
    if n == 1:
        return sorted_arr[0]
    idx = (p / 100.0) * (n - 1)
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return sorted_arr[low]
    frac = idx - low
    return sorted_arr[low] + frac * (sorted_arr[high] - sorted_arr[low])


def argsort(arr):
    """Returnează indicii care sortează crescător lista."""
    return sorted(range(len(arr)), key=lambda i: arr[i])


def argsort_desc(arr):
    """Returnează indicii care sortează descrescător lista."""
    return sorted(range(len(arr)), key=lambda i: arr[i], reverse=True)


def unique(arr):
    """Returnează valorile unice (ordonate)."""
    seen = set()
    result = []
    for x in arr:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return sorted(result)


def cumsum(arr):
    """Sumă cumulativă."""
    result = []
    s = 0.0
    for x in arr:
        s += x
        result.append(s)
    return result


def log1p_series(arr):
    """log(1 + x) element-cu-element."""
    return [math.log1p(x) for x in arr]


def expm1_series(arr):
    """exp(x) - 1 element-cu-element."""
    return [math.expm1(x) for x in arr]


def random_choice(arr, k, replace=False):
    """Alege k elemente din arr."""
    if replace:
        return [random.choice(arr) for _ in range(k)]
    else:
        return random.sample(arr, k)


def random_normal_vec(n):
    """Generează un vector cu n valori din N(0,1) folosind Box-Muller."""
    result = []
    while len(result) < n:
        u1 = random.random()
        u2 = random.random()
        if u1 == 0:
            continue
        mag = math.sqrt(-2.0 * math.log(u1))
        z1 = mag * math.cos(2.0 * math.pi * u2)
        z2 = mag * math.sin(2.0 * math.pi * u2)
        result.append(z1)
        if len(result) < n:
            result.append(z2)
    return result


def mat_from_list_of_lists(data):
    """Convertește o listă de liste într-o matrice de float-uri."""
    return [[float(x) for x in row] for row in data]


def mat_to_list_of_lists(A):
    """Convertește o matrice înapoi în listă de liste (deep copy)."""
    return [row[:] for row in A]
