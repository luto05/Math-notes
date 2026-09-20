"""Factorización QR por el proceso de Gram-Schmidt clásico.

Sección 10.4 de las Notas de Álgebra Lineal (teorema de existencia y unicidad).
Equivalente: ``np.linalg.qr``. Diferencia importante: numpy usa reflexiones de
Householder, numéricamente estables; el Gram-Schmidt clásico pierde
ortogonalidad cuando las columnas son casi dependientes (y numpy no exige
diagonal positiva en R, por lo que sus signos pueden diferir de estos).
"""
import math


def _col(A, j):
    return [fila[j] for fila in A]


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def factorizar_qr(A):
    """A (m×n, columnas linealmente independientes) = QR, con Q de columnas
    ortonormales y R triangular superior de diagonal positiva.

    Las fórmulas son las del teorema: r_jk = <a_k, q_j> para j < k y
    r_kk = ||v_k||, con v_k el vector de Gram-Schmidt."""
    m, n = len(A), len(A[0])
    qs = []  # columnas de Q
    R = [[0] * n for _ in range(n)]
    for k in range(n):
        a_k = _col(A, k)
        v = a_k[:]
        for j in range(k):
            R[j][k] = _dot(a_k, qs[j])
            v = [vi - R[j][k] * qj for vi, qj in zip(v, qs[j])]
        R[k][k] = math.sqrt(_dot(v, v))
        if R[k][k] < 1e-12:
            raise ValueError("columnas linealmente dependientes")
        qs.append([vi / R[k][k] for vi in v])
    Q = [[qs[j][i] for j in range(n)] for i in range(m)]
    return Q, R


def _tests():
    import random
    # el ejemplo de la sección 10.4
    A = [[1, 1], [1, 0], [0, 1]]
    Q, R = factorizar_qr(A)
    r2, r6 = math.sqrt(2), math.sqrt(6)
    esperado_R = [[r2, 1 / r2], [0, r6 / 2]]
    esperado_Q = [[1 / r2, 1 / r6], [1 / r2, -1 / r6], [0, 2 / r6]]
    for i in range(2):
        for j in range(2):
            assert abs(R[i][j] - esperado_R[i][j]) < 1e-12
    for i in range(3):
        for j in range(2):
            assert abs(Q[i][j] - esperado_Q[i][j]) < 1e-12
    # aleatorio: QR = A, Q^T Q = I, R triangular con diagonal positiva
    random.seed(2)
    for _ in range(20):
        m = random.randint(2, 6)
        n = random.randint(1, m)
        A = [[random.uniform(-5, 5) for _ in range(n)] for _ in range(m)]
        Q, R = factorizar_qr(A)
        for i in range(m):
            for j in range(n):
                qr = sum(Q[i][k] * R[k][j] for k in range(n))
                assert abs(qr - A[i][j]) < 1e-9
        for i in range(n):
            for j in range(n):
                qtq = _dot(_col(Q, i), _col(Q, j))
                assert abs(qtq - (1 if i == j else 0)) < 1e-9
                if i > j:
                    assert R[i][j] == 0
            assert R[i][i] > 0
    print("qr_gram_schmidt: ok")


if __name__ == "__main__":
    _tests()
