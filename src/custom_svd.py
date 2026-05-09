import numpy as np
import math

def house(A):
    R = np.copy(A).astype(float)
    m, n = np.shape(A)

    Q = np.eye(m)

    for k in range(0, n):
        V = np.copy(R[k:m, k])
        N = np.linalg.norm(V)

        if N > 0:
            sgn = np.sign(V[0]) if V[0] != 0 else 1.0
            V[0] = V[0] - sgn * N

            den = V.T @ V

            if abs(den) < 1e-12:
                H_v = np.eye(m - k)
            else:
                H_v = np.eye(m - k) - 2 * np.outer(V, V.T) / den
        else:
            H_v = np.eye(m - k)

        H = np.eye(m)
        H[k:m, k:m] = H_v

        R = H @ R
        Q = Q @ H

    return R, Q

def Tridiag_Householder(A):
    n = np.shape(A)[0] # Matricea este patratica
    T = np.copy(A)
    Q = np.eye(n)

    for k in range(0, n - 2):
        V = np.copy(T[k + 1:, k]) 
        N = np.linalg.norm(V)
        if N > 0:
            sgn = np.sign(V[0]) if V[0] != 0 else 1.0
            V[0] = V[0] - sgn * N
            I_mk = np.eye(n - k - 1)
            H_v = I_mk - 2 * np.outer(V, V.T) / (V.T @ V)
        else:
            H_v = np.eye(n - k - 1)
        H = np.eye(n)
        H[k+1:n, k+1:n] = H_v
        
        T = H @ T @ H
        Q = Q @ H
        
    return Q, T

def QR_iteration(A, Q, TOL=1e-6):
    T = Q.T @ A @ Q
    V = Q
    n=np.shape(A)[1]
    
    while np.linalg.norm(T - np.diag(np.diag(T)))**2 > TOL:
        R, q = house(T)
        T = R @ q
        V = V @ q

    return T, V

def SVD(A, TOL=1e-14):

    A = np.copy(A).astype(float)

    m = np.shape(A)[0]
    n = np.shape(A)[1]

    # Pasul 1: Tridiagonalizam A.T @ A
    B = A.T @ A
    Q0, T0 = Tridiag_Householder(B)

    # Pasul 2: Aflam valorile proprii ale lui A.T @ A
    D, V = QR_iteration(B, Q0)

    val = np.diag(D)

    # Pasul 3: Sortam valorile proprii descrescator
    indici = np.argsort(val)[::-1]

    val = val[indici]
    V = V[:, indici]

    val[val < 0] = 0
    sigma = np.sqrt(val)

    S = np.zeros((m, n))

    for i in range(n):
        S[i, i] = sigma[i]

    # Pasul 4: Vectorii singulari la stanga U
   
    U = np.zeros((m, m))

    coloana = 0 
    for i in range(n):
        if sigma[i] > TOL:
            U[:, i] = (A @ V[:, i]) / sigma[i]
            coloana += 1

    I = np.eye(m)
    for k in range(m): 
        if coloana >= m:
            break
            
        v = I[:, k] 
        w = np.copy(v)

        for j in range(coloana):
            w = w - (U[:, j].T @ v) * U[:, j]

        norma = np.linalg.norm(w)
        if norma > TOL:
            U[:, coloana] = w / norma 
            coloana += 1

    return U, S, V
