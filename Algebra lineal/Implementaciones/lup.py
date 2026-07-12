"""Factorización LUP y sus subproductos: sistemas, determinante e inversa.

Sección 10.3 de las Notas de Álgebra Lineal.
Equivalentes: ``np.linalg.solve``, ``np.linalg.det``, ``np.linalg.inv``,
``scipy.linalg.lu``. Diferencia: LAPACK factoriza por bloques y con análisis
de estabilidad fino; aquí hay solo pivoteo parcial por magnitud, como
recomienda el apunte. Funciona sobre cualquier cuerpo de Python (float,
Fraction), lo que permite tests exactos.
"""
from fractions import Fraction


def factorizar_lup(A):
    """Devuelve (L, U, perm) con PA = LU: L unitriangular inferior, U triangular
    superior y perm la permutación de filas (perm[k] = fila original en la
    posición k). En cada paso se pivotea por la entrada de mayor valor absoluto."""
    n = len(A)
    U = [fila[:] for fila in A]
    L = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    perm = list(range(n))
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(U[i][k]))
        if U[p][k] == 0:
            raise ValueError("matriz singular")
        if p != k:
            U[k], U[p] = U[p], U[k]
            perm[k], perm[p] = perm[p], perm[k]
            for j in range(k):  # los multiplicadores acompañan a sus filas
                L[k][j], L[p][j] = L[p][j], L[k][j]
        for i in range(k + 1, n):
            m = U[i][k] / U[k][k]  # multiplicador
            L[i][k] = m
            for j in range(k, n):  # complemento de Schur, fila i
                U[i][j] -= m * U[k][j]
    return L, U, perm


def sustitucion_adelante(L, b):
    """Resuelve Ly = b leyendo las ecuaciones en orden (L unitriangular)."""
    y = []
    for i in range(len(L)):
        y.append(b[i] - sum(L[i][j] * y[j] for j in range(i)))
    return y


def sustitucion_atras(U, y):
    """Resuelve Ux = y desde la última ecuación hacia la primera."""
    n = len(U)
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


def resolver(A, b):
    """Resuelve Ax = b vía PA = LU: equivale a L(Ux) = Pb, dos sustituciones."""
    L, U, perm = factorizar_lup(A)
    pb = [b[i] for i in perm]
    return sustitucion_atras(U, sustitucion_adelante(L, pb))


def determinante(A):
    """det(A) = (-1)^s por el producto de los pivotes (obs. de la sección 10.3)."""
    try:
        _, U, perm = factorizar_lup(A)
    except ValueError:
        return 0
    # signo de la permutación por su descomposición en ciclos
    s, vista = 0, [False] * len(perm)
    for i in range(len(perm)):
        j, largo = i, 0
        while not vista[j]:
            vista[j] = True
            j = perm[j]
            largo += 1
        s += max(largo - 1, 0)
    prod = 1
    for i in range(len(U)):
        prod *= U[i][i]
    return -prod if s % 2 else prod


def inversa(A):
    """Las columnas de la inversa son las soluciones de A x_j = e_j."""
    n = len(A)
    L, U, perm = factorizar_lup(A)
    cols = []
    for j in range(n):
        e = [1 if i == j else 0 for i in range(n)]
        pb = [e[i] for i in perm]
        cols.append(sustitucion_atras(U, sustitucion_adelante(L, pb)))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def _tests():
    import random
    F = Fraction
    # el ejemplo de la sección 10.3, exacto con fracciones
    A = [[F(1), F(2), F(0)], [F(3), F(4), F(4)], [F(5), F(6), F(3)]]
    b = [F(3), F(7), F(8)]
    assert resolver(A, b) == [F(-7, 5), F(11, 5), F(3, 5)]
    assert determinante(A) == 10
    Ai = inversa(A)
    for i in range(3):
        for j in range(3):
            assert sum(A[i][k] * Ai[k][j] for k in range(3)) == (1 if i == j else 0)
    # el ejemplo sin factorización LU del apunte: [[0,1],[1,1]] exige permutar
    assert resolver([[F(0), F(1)], [F(1), F(1)]], [F(2), F(3)]) == [F(1), F(2)]
    # PA = LU en la factorización con pivoteo
    L, U, perm = factorizar_lup(A)
    PA = [A[i] for i in perm]
    for i in range(3):
        for j in range(3):
            assert sum(L[i][k] * U[k][j] for k in range(3)) == PA[i][j]
    # sistemas aleatorios con flotantes: Ax ≈ b
    random.seed(1)
    for _ in range(30):
        n = random.randint(1, 6)
        M = [[random.uniform(-5, 5) for _ in range(n)] for _ in range(n)]
        c = [random.uniform(-5, 5) for _ in range(n)]
        x = resolver(M, c)
        for i in range(n):
            assert abs(sum(M[i][k] * x[k] for k in range(n)) - c[i]) < 1e-9
    print("lup: ok")


if __name__ == "__main__":
    _tests()
