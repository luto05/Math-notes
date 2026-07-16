"""Descenso de gradiente con paso fijo y promedio de iterados.

Sección 11.2 de las Notas de Álgebra Lineal (teorema de convergencia para
funciones convexas). Es el núcleo de los optimizadores de los frameworks de
aprendizaje automático, que usan variantes: gradiente estocástico por
minibatches, momentum, pasos adaptativos (Adam); aquí el paso es fijo y se
devuelve el promedio x̄, que es la variante con garantía del teorema.
"""


def descenso_gradiente(gradiente, x0, eta, T):
    """Itera x <- x - eta * gradiente(x) durante T pasos.

    Devuelve (promedio, iterados): el promedio de x^(0), ..., x^(T-1) —el x̄
    del teorema, con f(x̄) - f(x*) <= RL/sqrt(T) para la eta óptima— y la
    lista completa de iterados x^(0), ..., x^(T)."""
    iterados = [list(x0)]
    x = list(x0)
    for _ in range(T):
        g = gradiente(x)
        x = [xi - eta * gi for xi, gi in zip(x, g)]
        iterados.append(x)
    n = len(x0)
    promedio = [sum(it[i] for it in iterados[:T]) / T for i in range(n)]
    return promedio, iterados


def _tests():
    from fractions import Fraction as F
    # el ejemplo de la sección 11.2: f(x) = (1/2)||Ax - b||^2 con la A y b de
    # mínimos cuadrados; gradiente A^T A x - A^T b, x0 = 0, eta = 1/4.
    # Iterados exactos: (1/2, 1/2), (5/8, 5/8), (21/32, 21/32) -> (2/3, 2/3)
    def gradiente(x):
        return [2 * x[0] + x[1] - 2, x[0] + 2 * x[1] - 2]

    _, its = descenso_gradiente(gradiente, [F(0), F(0)], F(1, 4), 3)
    assert its[1] == [F(1, 2), F(1, 2)]
    assert its[2] == [F(5, 8), F(5, 8)]
    assert its[3] == [F(21, 32), F(21, 32)]
    # forma cerrada del ejemplo: x^(t) = 2/3 - (2/3) 4^(-t) en cada coordenada
    for t, it in enumerate(its):
        assert it[0] == F(2, 3) - F(2, 3) / 4 ** t
    # con muchas iteraciones, f decrece hacia f(x̂) = 1/6
    def f(x):
        r = [x[0] + x[1] - 1, x[0] - 1, x[1] - 1]
        return sum(ri ** 2 for ri in r) / 2

    _, its = descenso_gradiente(gradiente, [0.0, 0.0], 0.25, 60)
    valores = [f(x) for x in its]
    assert all(a >= b - 1e-15 for a, b in zip(valores, valores[1:]))
    assert abs(valores[-1] - 1 / 6) < 1e-12
    print("descenso_gradiente: ok")


if __name__ == "__main__":
    _tests()
