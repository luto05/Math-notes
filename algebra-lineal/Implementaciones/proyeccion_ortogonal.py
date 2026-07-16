"""Proyección ortogonal sobre el espacio columna de una matriz.

Secciones 2 (teoría de la proyección) y 10.4 (la identidad P = QQ^T) de las
Notas de Álgebra Lineal. No tiene un equivalente directo de una llamada en
numpy; es el bloque con el que se construyen los mínimos cuadrados.
"""
from qr_gram_schmidt import factorizar_qr, _col, _dot


def proyectar(A, b):
    """P_{Col(A)}(b) = Q Q^T b, con A = QR la factorización de Gram-Schmidt."""
    Q, _ = factorizar_qr(A)
    n = len(Q[0])
    coeficientes = [_dot(_col(Q, j), b) for j in range(n)]  # Q^T b
    return [sum(coeficientes[j] * Q[i][j] for j in range(n)) for i in range(len(Q))]


def _tests():
    import random
    # el ejemplo de la sección 10.4: proyección (4/3, 2/3, 2/3) y residuo ⊥ columnas
    A = [[1, 1], [1, 0], [0, 1]]
    b = [1, 1, 1]
    p = proyectar(A, b)
    esperado = [4 / 3, 2 / 3, 2 / 3]
    assert all(abs(pi - ei) < 1e-12 for pi, ei in zip(p, esperado))
    residuo = [bi - pi for bi, pi in zip(b, p)]
    for j in range(2):
        assert abs(_dot(_col(A, j), residuo)) < 1e-12
    # idempotencia: proyectar dos veces es proyectar una
    random.seed(4)
    for _ in range(20):
        m, n = 5, 2
        A = [[random.uniform(-5, 5) for _ in range(n)] for _ in range(m)]
        b = [random.uniform(-5, 5) for _ in range(m)]
        p1 = proyectar(A, b)
        p2 = proyectar(A, p1)
        assert all(abs(a - c) < 1e-9 for a, c in zip(p1, p2))
    print("proyeccion_ortogonal: ok")


if __name__ == "__main__":
    _tests()
