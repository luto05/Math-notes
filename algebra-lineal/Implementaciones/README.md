# Implementaciones — Notas de Álgebra Lineal

Implementaciones en **Python puro** (sin numpy, solo biblioteca estándar) de los algoritmos de las secciones computacionales de [las notas](../notas-de-algebra-lineal.pdf). La idea es mostrar lo que `numpy`/`sklearn` hacen por dentro: cada archivo implementa la construcción exacta del apunte y se testea con los **mismos ejemplos verificados a mano** que aparecen en él.

## Cómo correrlas

Cada archivo trae sus tests en un bloque `_tests()`; correrlo los ejecuta:

```bash
cd Implementaciones
python3 lup.py            # imprime "lup: ok" si todos los asserts pasan
for f in *.py; do python3 "$f"; done   # la suite completa
```

Para usar una función, importarla (desde esta carpeta, porque algunos archivos se importan entre sí):

```python
from lup import resolver
x = resolver([[1, 2, 0], [3, 4, 4], [5, 6, 3]], [3, 7, 8])
# con fracciones exactas: from fractions import Fraction, y x == [-7/5, 11/5, 3/5]
```

Las matrices son listas de filas (`list[list]`); las funciones aceptan `float` o `Fraction` (los tests usan fracciones donde el ejemplo del apunte es exacto).

## Archivos

| Archivo | Contenido | Equivalente | Sección |
|---|---|---|---|
| `producto_bloques.py` | multiplicación directa y por bloques 2×2 recursiva | `A @ B` | 10.1–10.2 |
| `lup.py` | factorización PA = LU con pivoteo parcial, sustituciones, `resolver`, `determinante`, `inversa` | `np.linalg.solve/det/inv`, `scipy.linalg.lu` | 10.3 |
| `qr_gram_schmidt.py` | factorización A = QR por Gram-Schmidt clásico | `np.linalg.qr` | 10.4 |
| `minimos_cuadrados.py` | ecuaciones normales y la vía QR ($R\hat x = Q^Tb$) | `np.linalg.lstsq` | 10.4 |
| `proyeccion_ortogonal.py` | proyección sobre Col(A) vía $QQ^T$ | — | 2 y 10.4 |
| `metodo_potencia.py` | autovalor dominante con cociente de Rayleigh; deflación para los primeros k de una simétrica | `np.linalg.eigh` | 10.5 |
| `pca.py` | componentes principales vía covarianza + potencia con deflación | `sklearn.decomposition.PCA` | 9 y 10.5 |
| `kmeans_lloyd.py` | procedimiento de Lloyd con historial de costos | `sklearn.cluster.KMeans` | 11.1 |
| `descenso_gradiente.py` | paso fijo y promedio de iterados (la variante del teorema de convergencia) | el núcleo de todo optimizador de ML | 11.2 |

## Diferencias con numpy / LAPACK / sklearn

Estas implementaciones son **didácticas**: fieles al apunte y correctas, pero no un reemplazo de las bibliotecas numéricas. Las diferencias concretas:

- **Rendimiento.** numpy delega en BLAS/LAPACK: bloques ajustados a la caché, instrucciones vectoriales y paralelismo. Estas versiones, con listas de Python, son órdenes de magnitud más lentas; el costo *asintótico* es el mismo.
- **QR.** `np.linalg.qr` usa reflexiones de Householder, numéricamente estables. El Gram-Schmidt clásico del apunte pierde ortogonalidad cuando las columnas son casi dependientes. Además, numpy no exige diagonal positiva en $R$, así que sus signos pueden diferir de los de acá (la factorización con diagonal positiva es la única, según el teorema de unicidad).
- **Autovalores.** `np.linalg.eigh` usa el algoritmo QR con desplazamientos: obtiene todo el espectro a la vez y tolera autovalores repetidos. El método de la potencia requiere $\lambda_1 > |\lambda_2|$, converge con razón $|\lambda_2|/\lambda_1$ (lenta si están cerca) y la deflación acumula error de un par al siguiente. A su favor: es lo que se usa cuando la matriz es enorme y dispersa y solo se necesitan las primeras componentes (PageRank).
- **PCA.** sklearn no forma la matriz de covarianzas: hace la SVD de los datos centrados, que evita amplificar los errores de redondeo (es la observación «PCA mediante SVD» del apunte). Acá se forma $C = X^TX/(m-1)$ porque esa es la construcción con la que el apunte lo define.
- **Mínimos cuadrados.** `np.linalg.lstsq` resuelve vía SVD y maneja matrices de rango deficiente; acá se exigen columnas linealmente independientes (la hipótesis del teorema).
- **k-means.** sklearn inicializa con k-means++ y corre múltiples reinicios para esquivar mínimos locales malos; acá la inicialización es explícita y se corre una vez, como el procedimiento de Lloyd del apunte.
- **Descenso de gradiente.** Paso fijo y promedio de iterados, que es la variante con garantía demostrada ($RL/\sqrt{T}$); los frameworks usan gradiente estocástico por minibatches y variantes adaptativas (momentum, Adam), sin esa garantía pero mucho más rápidas en la práctica.
- **LUP.** Hay pivoteo parcial por magnitud (el del apunte), pero no el análisis de estabilidad, escalado ni la organización por bloques de LAPACK. A cambio, funciona sobre `Fraction`, lo que permite tests *exactos*.
