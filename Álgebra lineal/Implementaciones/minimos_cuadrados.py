"""Mínimos cuadrados: por ecuaciones normales y por la vía QR.

Sección 10.4 de las Notas de Álgebra Lineal (teorema de las ecuaciones
normales y la observación sobre QR).
Equivalente: ``np.linalg.lstsq``. Diferencias: numpy resuelve con una SVD y
maneja rango deficiente; aquí se exigen columnas linealmente independientes.
La vía por QR evita formar A^T A (mejor condicionada), la de las ecuaciones
normales se incluye porque es la del teorema.
"""
from lup import resolver
from qr_gram_schmidt import factorizar_qr, _col, _dot


def ecuaciones_normales(A, b):
    """Resuelve A^T A x = A^T b (Teorema «Ecuaciones Normales»)."""
    n = len(A[0])
    AtA = [[_dot(_col(A, i), _col(A, j)) for j in range(n)] for i in range(n)]
    Atb = [_dot(_col(A, i), b) for i in range(n)]
    return resolver(AtA, Atb)


def minimos_cuadrados(A, b):
    """Vía QR: R x = Q^T b, resuelto por sustitución hacia atrás."""
    Q, R = factorizar_qr(A)
    n = len(R)
    qtb = [_dot(_col(Q, i), b) for i in range(n)]
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = (qtb[i] - sum(R[i][j] * x[j] for j in range(i + 1, n))) / R[i][i]
    return x


def _tests():
    import random
    # el ejemplo de la sección 10.4: x̂ = (2/3, 2/3), residuo (−1/3, 1/3, 1/3)
    A = [[1, 1], [1, 0], [0, 1]]
    b = [1, 1, 1]
    for metodo in (ecuaciones_normales, minimos_cuadrados):
        x = metodo(A, b)
        assert abs(x[0] - 2 / 3) < 1e-12 and abs(x[1] - 2 / 3) < 1e-12
    # el residuo es ortogonal a las columnas de A (caracterización de la proyección)
    x = minimos_cuadrados(A, b)
    residuo = [b[i] - sum(A[i][j] * x[j] for j in range(2)) for i in range(3)]
    for j in range(2):
        assert abs(_dot(_col(A, j), residuo)) < 1e-12
    # aleatorio: ambas vías coinciden
    random.seed(3)
    for _ in range(20):
        m = random.randint(3, 7)
        n = random.randint(1, m - 1)
        A = [[random.uniform(-5, 5) for _ in range(n)] for _ in range(m)]
        b = [random.uniform(-5, 5) for _ in range(m)]
        x1, x2 = ecuaciones_normales(A, b), minimos_cuadrados(A, b)
        assert all(abs(a - c) < 1e-8 for a, c in zip(x1, x2))
    print("minimos_cuadrados: ok")


if __name__ == "__main__":
    _tests()
