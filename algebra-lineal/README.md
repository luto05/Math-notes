# Notas de Álgebra Lineal

Un recorrido completo por el álgebra lineal, de la definición de espacio vectorial al aprendizaje automático: espacios con producto interno y proyecciones ortogonales, transformaciones lineales y representación matricial, el espacio dual, determinantes (existencia y unicidad como función multilineal alternada), autovalores y el teorema espectral, formas cuadráticas, SVD y PCA. Las dos últimas secciones tratan las matrices como objetos computacionales (representación en memoria, producto por bloques, factorizaciones LU/LUP y QR, mínimos cuadrados) y dos algoritmos de aprendizaje automático (k-means con el procedimiento de Lloyd y el descenso de gradiente con su garantía de convergencia).

El PDF compilado (83 páginas) está incluido: [`notas-de-algebra-lineal.pdf`](notas-de-algebra-lineal.pdf).

## Secciones

1. Espacios Vectoriales y Subespacios
2. Espacios Ortogonales, Independencia Lineal, Base y Dimensión
3. Transformaciones Lineales
4. El Espacio Dual
5. Determinantes
6. Autovalores, Autovectores y Diagonalización
7. Aplicación: Optimización de Formas Cuadráticas
8. Valores Singulares: Generalización de la Diagonalización
9. Análisis de Componentes Principales
10. Matrices en la Computadora (con el método de la potencia)
11. Algoritmos de Aprendizaje Automático

## Implementaciones

Los algoritmos de las secciones computacionales (producto por bloques, LUP, QR, mínimos cuadrados, método de la potencia, PCA, k-means, descenso de gradiente) están implementados en Python puro en [`Implementaciones/`](Implementaciones/), con tests que reproducen los ejemplos del documento; su README explica cómo correrlas y qué las diferencia de numpy/sklearn.

## Compilación

```bash
lualatex notas-de-algebra-lineal.tex   # dos pasadas para resolver referencias
lualatex notas-de-algebra-lineal.tex
```

Requiere LuaLaTeX y el paquete `estilo-mate.sty` (véase el README de la raíz del repositorio).
