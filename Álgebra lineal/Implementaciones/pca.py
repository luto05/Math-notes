"""Análisis de Componentes Principales por potencia + deflación.

Secciones 9 (PCA) y 10.5 (método de la potencia) de las Notas de Álgebra
Lineal. Equivalente: ``sklearn.decomposition.PCA``. Diferencias: sklearn
calcula una SVD de la matriz de datos centrada, sin formar la covarianza
(numéricamente preferible, como señala la sección de PCA vía SVD); aquí se
forma C = X^T X / (m-1) y se extraen sus autovectores dominantes con el
método de la potencia, que es la construcción del apunte.
"""
from metodo_potencia import autopares_simetrica


def pca(datos, k, iteraciones=2000):
    """Devuelve (componentes, varianzas, medias): las primeras k componentes
    principales (como filas), la varianza capturada por cada una (autovalor de
    la matriz de covarianzas) y la media por columna usada para centrar."""
    m, n = len(datos), len(datos[0])
    medias = [sum(fila[j] for fila in datos) / m for j in range(n)]
    X = [[fila[j] - medias[j] for j in range(n)] for fila in datos]
    C = [[sum(X[t][i] * X[t][j] for t in range(m)) / (m - 1) for j in range(n)]
         for i in range(n)]
    pares = autopares_simetrica(C, k, iteraciones)
    componentes = [v for _, v in pares]
    varianzas = [lam for lam, _ in pares]
    return componentes, varianzas, medias


def _tests():
    import math
    # datos centrados con covarianza [[20/3, 16/3], [16/3, 20/3]]:
    # autovalores 12 y 4/3, componentes (1,1)/√2 y (1,-1)/√2
    datos = [(3, 3), (-3, -3), (1, -1), (-1, 1)]
    componentes, varianzas, medias = pca(datos, 2)
    r2 = math.sqrt(2)
    assert all(abs(mu) < 1e-12 for mu in medias)
    assert abs(varianzas[0] - 12) < 1e-9
    assert abs(varianzas[1] - 4 / 3) < 1e-9
    c1, c2 = componentes
    assert abs(abs(c1[0]) - 1 / r2) < 1e-9 and abs(c1[0] - c1[1]) < 1e-9
    assert abs(abs(c2[0]) - 1 / r2) < 1e-9 and abs(c2[0] + c2[1]) < 1e-9
    # la suma de varianzas es la traza de la covarianza (40/3)
    assert abs(sum(varianzas) - 40 / 3) < 1e-9
    print("pca: ok")


if __name__ == "__main__":
    _tests()
