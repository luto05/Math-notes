"""Método de la potencia con cociente de Rayleigh, y deflación para simétricas.

Sección 10.5 de las Notas de Álgebra Lineal.
Equivalente: ``np.linalg.eigh``. Diferencias: LAPACK usa el algoritmo QR con
desplazamientos, que calcula todo el espectro a la vez y tolera autovalores
repetidos; la potencia obtiene el dominante (y, con deflación, los primeros k
de una simétrica), converge geométricamente con razón |λ2|/λ1 y requiere
λ1 > |λ2|.
"""
import math
import random


def _matvec(A, x):
    return [sum(a * xi for a, xi in zip(fila, x)) for fila in A]


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def _normalizar(v):
    norma = math.sqrt(_dot(v, v))
    return [vi / norma for vi in v]


def metodo_potencia(A, iteraciones=1000, x0=None, semilla=0):
    """Devuelve (lambda, v): el autovalor dominante estimado por el cociente de
    Rayleigh y su autovector unitario. Un x0 aleatorio tiene componente no nula
    sobre v1 con probabilidad 1 (obs. de la sección 10.5)."""
    n = len(A)
    if x0 is None:
        rng = random.Random(semilla)
        x0 = [rng.uniform(-1, 1) for _ in range(n)]
    x = _normalizar(x0)
    for _ in range(iteraciones):
        x = _normalizar(_matvec(A, x))
    return _dot(x, _matvec(A, x)), x


def autopares_simetrica(A, k, iteraciones=1000, semilla=0):
    """Los k pares (autovalor, autovector) de mayor autovalor de una matriz
    simétrica, por potencia + deflación A' = A - lambda v v^T."""
    A = [fila[:] for fila in A]
    pares = []
    for _ in range(k):
        lam, v = metodo_potencia(A, iteraciones, semilla=semilla)
        pares.append((lam, v))
        for i in range(len(A)):
            for j in range(len(A)):
                A[i][j] -= lam * v[i] * v[j]
    return pares


def _tests():
    from fractions import Fraction
    # el ejemplo de la sección 10.5: A = [[2,1],[1,2]], autovalores 3 y 1
    A = [[2, 1], [1, 2]]
    # iterados sin normalizar desde (1,0): (2,1), (5,4), (14,13), (41,40)
    x = [1, 0]
    for esperado in ([2, 1], [5, 4], [14, 13], [41, 40]):
        x = _matvec(A, x)
        assert x == esperado
    # cocientes de Rayleigh exactos del ejemplo: 2, 14/5, 122/41, 1094/365
    x = [1, 0]
    rayleighs = []
    for _ in range(4):
        rayleighs.append(Fraction(_dot(x, _matvec(A, x)), _dot(x, x)))
        x = _matvec(A, x)
    assert rayleighs == [Fraction(2), Fraction(14, 5), Fraction(122, 41), Fraction(1094, 365)]
    # convergencia: lambda ≈ 3, v ≈ ±(1,1)/√2; deflación: el segundo par ≈ (1, ±(1,-1)/√2)
    (l1, v1), (l2, v2) = autopares_simetrica(A, 2, iteraciones=200)
    r2 = math.sqrt(2)
    assert abs(l1 - 3) < 1e-9 and abs(abs(v1[0]) - 1 / r2) < 1e-9 and abs(v1[0] - v1[1]) < 1e-9
    assert abs(l2 - 1) < 1e-9 and abs(abs(v2[0]) - 1 / r2) < 1e-9 and abs(v2[0] + v2[1]) < 1e-9
    # simétricas 2x2 aleatorias contra la fórmula cuadrática
    rng = random.Random(5)
    for _ in range(20):
        a, c = rng.uniform(1, 5), rng.uniform(1, 5)
        b = rng.uniform(0.1, 0.9)
        M = [[a, b], [b, c]]
        esperado = ((a + c) + math.sqrt((a - c) ** 2 + 4 * b * b)) / 2
        lam, v = metodo_potencia(M, iteraciones=2000)
        assert abs(lam - esperado) < 1e-9
        # A v ≈ lambda v
        Av = _matvec(M, v)
        assert all(abs(avi - lam * vi) < 1e-6 for avi, vi in zip(Av, v))
    print("metodo_potencia: ok")


if __name__ == "__main__":
    _tests()
