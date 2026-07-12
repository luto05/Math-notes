"""Multiplicación de matrices: directa y por bloques (divide y vencerás).

Secciones 10.1 y 10.2 de las Notas de Álgebra Lineal.
Equivalente en numpy: ``A @ B``. Diferencia: numpy delega en BLAS, que usa
bloques ajustados a la caché, instrucciones vectoriales y paralelismo; el
costo asintótico es el mismo, la constante no.
"""


def multiplicar(A, B):
    """Producto directo por la definición, Theta(n·m·p) operaciones."""
    n, m, p = len(A), len(B), len(B[0])
    assert len(A[0]) == m, "dimensiones incompatibles"
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            s = 0
            for k in range(m):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def _sumar(A, B):
    return [[a + b for a, b in zip(fa, fb)] for fa, fb in zip(A, B)]


def _bloque(A, fi, ff, ci, cf):
    return [fila[ci:cf] for fila in A[fi:ff]]


def _ensamblar(C11, C12, C21, C22):
    arriba = [f1 + f2 for f1, f2 in zip(C11, C12)]
    abajo = [f1 + f2 for f1, f2 in zip(C21, C22)]
    return arriba + abajo


def multiplicar_bloques(A, B):
    """Producto por bloques 2x2, recursivo (Proposición «Producto por Bloques»).

    Igual cantidad de operaciones que el directo: T(n) = 8 T(n/2) + Theta(1)
    da Theta(n^3); el interés es que esta partición es la base de Strassen.
    """
    n = len(A)
    if n <= 2 or n % 2 == 1:
        return multiplicar(A, B)
    h = n // 2
    A11, A12 = _bloque(A, 0, h, 0, h), _bloque(A, 0, h, h, n)
    A21, A22 = _bloque(A, h, n, 0, h), _bloque(A, h, n, h, n)
    B11, B12 = _bloque(B, 0, h, 0, h), _bloque(B, 0, h, h, n)
    B21, B22 = _bloque(B, h, n, 0, h), _bloque(B, h, n, h, n)
    m = multiplicar_bloques
    C11 = _sumar(m(A11, B11), m(A12, B21))
    C12 = _sumar(m(A11, B12), m(A12, B22))
    C21 = _sumar(m(A21, B11), m(A22, B21))
    C22 = _sumar(m(A21, B12), m(A22, B22))
    return _ensamblar(C11, C12, C21, C22)


def _tests():
    import random
    random.seed(0)
    # bloques == directo sobre matrices aleatorias enteras (comparación exacta)
    for _ in range(20):
        n = random.choice([2, 4, 8])
        A = [[random.randint(-9, 9) for _ in range(n)] for _ in range(n)]
        B = [[random.randint(-9, 9) for _ in range(n)] for _ in range(n)]
        assert multiplicar(A, B) == multiplicar_bloques(A, B)
    # rectangular, contra el cálculo a mano
    A = [[1, 2], [3, 4], [5, 6]]
    B = [[1, 0, 2], [0, 1, 3]]
    assert multiplicar(A, B) == [[1, 2, 8], [3, 4, 18], [5, 6, 28]]
    print("producto_bloques: ok")


if __name__ == "__main__":
    _tests()
