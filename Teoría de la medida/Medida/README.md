# Apuntes de Teoría de la Medida

De la integral de Riemann a la de Lebesgue y de ahí a la probabilidad moderna: σ-álgebras y medidas (con el ejemplo de Vitali), funciones medibles y convergencia, integración con sus teoremas de convergencia, espacios L^p, producto de medidas (Fubini-Tonelli), Radon-Nikodym, esperanza condicional, martingalas y teoremas de Doob, movimiento browniano, integral de Itô, Girsanov y, como aplicación final, el modelo de Black-Scholes.

El PDF compilado (77 páginas) está incluido: [`Apuntes de Teoría de la Medida.pdf`](<Apuntes de Teoría de la Medida.pdf>).

## Secciones

1. Motivación: de Riemann a Lebesgue
2. σ-álgebras
3. Conjuntos no medibles: el ejemplo de Vitali
4. Medidas
5. Generación de medidas
6. Funciones medibles
7. Convergencia de funciones medibles
8. Integración
9. Teoremas de convergencia
10. Los espacios L^p
11. Producto de medidas: Fubini-Tonelli
12. Espacios de medida completos
13. Medidas con signo y el teorema de Radon-Nikodym
14. Esperanza Condicional
15. Martingalas
16. Teoremas de Doob
17. Movimiento Browniano
18. Integral de Itô
19. Teorema de Girsanov
20. Aplicación: el modelo de Black-Scholes

## Compilación

```bash
lualatex "Apuntes de Teoría de la Medida.tex"   # dos pasadas para resolver referencias
lualatex "Apuntes de Teoría de la Medida.tex"
```

Requiere LuaLaTeX y el paquete `estilo-mate.sty` (véase el README de la raíz del repositorio).
