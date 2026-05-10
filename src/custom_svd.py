import numpy as np


def Tridiag_Householder(A):
    n = A.shape[0]
    T = np.copy(A).astype(float)
    Q = np.eye(n)

    for k in range(n - 2):
        x = T[k + 1:, k].copy()
        norm_x = np.linalg.norm(x)

        if abs(norm_x) < 1e-14:
            continue

        sgn = np.sign(x[0]) if abs(x[0]) > 1e-14 else 1.0
        x[0] += sgn * norm_x
        v = x / np.linalg.norm(x)

        # Apply Householder reflection from the left:  T = H @ T
        T[k + 1:, :] -= 2 * np.outer(v, v @ T[k + 1:, :])

        # Apply Householder reflection from the right: T = T @ H
        T[:, k + 1:] -= 2 * np.outer(T[:, k + 1:] @ v, v)

        # Accumulate orthogonal transformations
        Q[:, k + 1:] -= 2 * np.outer(Q[:, k + 1:] @ v, v)

    # Force exact tridiagonal symmetry (clean numerical noise)
    T = np.tril(np.triu(T, -1), 1)
    T = (T + T.T) / 2

    return Q, T


def _qr_step_tridiag_explicit(T_block):
    """
    One explicit QR step with Wilkinson shift on a symmetric
    tridiagonal matrix represented as a dense block.
    Uses Givens rotations.  Returns the updated block and
    the list of rotations (c, s) for each step k.
    """
    n = T_block.shape[0]
    T_shifted = T_block.copy()

    # Wilkinson shift
    delta = (T_shifted[n - 2, n - 2] - T_shifted[n - 1, n - 1]) / 2.0
    sign = np.sign(delta) if abs(delta) > 1e-14 else 1.0
    mu = T_shifted[n - 1, n - 1] - T_shifted[n - 2, n - 1] ** 2 / (
        delta + sign * np.sqrt(delta ** 2 + T_shifted[n - 2, n - 1] ** 2)
    )
    T_shifted -= mu * np.eye(n)

    rotations = []

    # Left rotations: build R = Q^T (T - mu*I)
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

    # Right rotations: T' = R @ Q + mu*I  (Q = G0 G1 ...)
    for k in range(n - 1):
        c, s = rotations[k]
        col_k = T_shifted[:, k].copy()
        col_k1 = T_shifted[:, k + 1].copy()
        T_shifted[:, k] = c * col_k + s * col_k1
        T_shifted[:, k + 1] = -s * col_k + c * col_k1

    T_new = T_shifted + mu * np.eye(n)

    # Clean numerical noise below tridiagonal band
    T_new = np.tril(np.triu(T_new, -1), 1)
    T_new = (T_new + T_new.T) / 2

    return T_new, rotations


def QR_iteration(A, Q, TOL=1e-6):
    """
    Computes eigenvalues/vectors of symmetric matrix A using the
    tridiagonal form Q.T @ A @ Q produced by Tridiag_Householder.
    Uses explicit Givens QR steps with Wilkinson shift and deflation.
    """
    T = Q.T @ A @ Q
    n = A.shape[0]
    V = Q.copy()

    active_end = n
    total_iters = 0
    max_total_iters = 5000

    while active_end > 1 and total_iters < max_total_iters:
        # Deflate converged bottom elements
        while active_end > 1 and abs(T[active_end - 1, active_end - 2]) < TOL:
            active_end -= 1

        if active_end <= 1:
            break

        # Run QR steps on the active block until bottom off-diagonal converges
        block_iters = 0
        while abs(T[active_end - 1, active_end - 2]) >= TOL and block_iters < 10:
            T_block = T[:active_end, :active_end].copy()
            T_block_new, rots = _qr_step_tridiag_explicit(T_block)
            T[:active_end, :active_end] = T_block_new

            # Accumulate rotations into eigenvector matrix
            for k, (c, s) in enumerate(rots):
                v_k = V[:, k].copy()
                v_k1 = V[:, k + 1].copy()
                V[:, k] = c * v_k + s * v_k1
                V[:, k + 1] = -s * v_k + c * v_k1

            total_iters += 1
            block_iters += 1

        # Force exact zero on the deflated off-diagonal
        if active_end > 1:
            T[active_end - 1, active_end - 2] = 0.0
            T[active_end - 2, active_end - 1] = 0.0
            active_end -= 1

    D = np.diag(np.diag(T))
    return D, V


def SVD(A, TOL=1e-14):
    A = np.copy(A).astype(float)

    m = A.shape[0]
    n = A.shape[1]

    # Pasul 1: Tridiagonalizam A.T @ A (manual - Householder)
    B = A.T @ A
    Q0, T0 = Tridiag_Householder(B)

    # Pasul 2: Aflam valorile proprii ale lui A.T @ A (QR manual cu deflatie)
    D, V = QR_iteration(B, Q0)

    val = np.diag(D)

    # Pasul 3: Sortam valorile proprii descrescator
    indici = np.argsort(val)[::-1]

    val = val[indici]
    V = V[:, indici]

    val[val < 0] = 0
    sigma = np.sqrt(val)

    S = np.zeros((n, n))

    for i in range(n):
        S[i, i] = sigma[i]

    # Pasul 4: Vectorii singulari la stanga U (economy size m x n)
    U = np.zeros((m, n))

    for i in range(n):
        if sigma[i] > TOL:
            U[:, i] = (A @ V[:, i]) / sigma[i]

    return U, S, V
