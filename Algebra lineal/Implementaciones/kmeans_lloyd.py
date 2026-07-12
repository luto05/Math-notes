"""k-means con el procedimiento de Lloyd.

Sección 11.1 de las Notas de Álgebra Lineal.
Equivalente: ``sklearn.cluster.KMeans``. Diferencias: sklearn inicializa con
k-means++ y corre varios reinicios para esquivar mínimos locales malos; aquí
los centros iniciales se dan explícitamente (o se muestrean al azar) y se
corre una sola vez, que es el procedimiento del apunte.
"""
import random


def _delta(x, c):
    """Distancia cuadrática ||x - c||^2."""
    return sum((xi - ci) ** 2 for xi, ci in zip(x, c))


def _centroide(puntos):
    m = len(puntos)
    return tuple(sum(p[j] for p in puntos) / m for j in range(len(puntos[0])))


def kmeans(puntos, k, centros=None, semilla=0):
    """Procedimiento de Lloyd. Devuelve (clusters, centros, historial):
    clusters como listas de índices de los puntos, y el costo f(S, C) tras
    cada recálculo de centroides — decreciente y finito por la Proposición
    «Monotonía y Terminación de Lloyd». Los empates de distancia se rompen
    por el índice menor del centro (la regla fija del apunte)."""
    rng = random.Random(semilla)
    centros = [tuple(c) for c in (centros if centros is not None else rng.sample(list(puntos), k))]
    historial = []
    asignacion_anterior = None
    while True:
        # asignar cada punto a un centro más cercano
        asignacion = [min(range(k), key=lambda l: _delta(p, centros[l])) for p in puntos]
        if asignacion == asignacion_anterior:
            break
        asignacion_anterior = asignacion
        # recalcular cada centro como el centroide de su cluster
        for l in range(k):
            grupo = [p for p, a in zip(puntos, asignacion) if a == l]
            if grupo:
                centros[l] = _centroide(grupo)
        historial.append(sum(_delta(p, centros[a]) for p, a in zip(puntos, asignacion)))
    clusters = [[i for i, a in enumerate(asignacion) if a == l] for l in range(k)]
    return clusters, centros, historial


def _tests():
    # el ejemplo de la sección 11.1: S = {0, 2, 5, 6}, centros iniciales 0 y 2;
    # el punto 2 cambia de cluster y el costo baja 26/3 -> 5/2 (óptimo)
    puntos = [(0,), (2,), (5,), (6,)]
    clusters, centros, historial = kmeans(puntos, 2, centros=[(0,), (2,)])
    assert clusters == [[0, 1], [2, 3]]
    assert centros == [(1.0,), (5.5,)]
    assert len(historial) == 2
    assert abs(historial[0] - 26 / 3) < 1e-12
    assert abs(historial[1] - 5 / 2) < 1e-12
    # aleatorio: el costo nunca aumenta entre recálculos
    rng = random.Random(6)
    for _ in range(10):
        pts = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(30)]
        _, _, hist = kmeans(pts, 3, semilla=7)
        assert all(a >= b - 1e-9 for a, b in zip(hist, hist[1:]))
    print("kmeans_lloyd: ok")


if __name__ == "__main__":
    _tests()
