# Matemática — Apuntes

Apuntes de matemática en LaTeX, con demostraciones completas: álgebra lineal, análisis real y teoría de la medida, desde los espacios vectoriales y la topología de espacios métricos hasta la integral de Lebesgue, las martingalas y el modelo de Black-Scholes.

## Contenido

| Documento | Páginas | Temas |
|---|---|---|
| [`algebra-lineal/`](algebra-lineal/) | 83 | Espacios vectoriales, producto interno y proyecciones, transformaciones lineales, dualidad, determinantes, diagonalización y teorema espectral, SVD y PCA, factorizaciones LU/QR, mínimos cuadrados y método de la potencia, k-means y descenso de gradiente. Incluye implementaciones en Python puro de la parte computacional |
| [`analisis-real/`](analisis-real/) | 85 | Cardinalidad, la recta real, espacios métricos, compacidad y continuidad, convexidad, teoremas de separación |
| [`teoria-de-la-medida/Medida/`](teoria-de-la-medida/Medida/) | 77 | De Riemann a Lebesgue, σ-álgebras y medidas, integración y teoremas de convergencia, espacios L^p, Fubini-Tonelli, Radon-Nikodym, martingalas, movimiento browniano, integral de Itô y Black-Scholes |

Cada carpeta incluye el PDF compilado y un README con el detalle de secciones.

## Requisitos

- **LuaLaTeX** (TeX Live reciente) para recompilar; los PDF compilados están incluidos.
- **Python 3** (solo biblioteca estándar) para las implementaciones de `algebra-lineal/Implementaciones/`.
- El paquete propio `estilo-mate.sty` (cajas de teoremas, tipografía y layout), instalado donde TeX lo encuentre (p. ej. `~/texmf/tex/latex/estilo-mate/`). Dos pasadas resuelven las referencias cruzadas.
