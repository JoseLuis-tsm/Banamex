# Fase 1 — Datos: hallazgos y decisiones

> Todas las cifras salen de los scripts de `scripts/01_datos/` corridos sobre `data/raw/base_cards_sintetica.csv` (base **sintética**). Para regenerarlas: `python scripts/run_fase.py 1`. Las tablas completas quedan en `reports/01_datos/`.

## 1. Formato del CSV (preguntas abiertas 1 y 2 de CLAUDE.md §12)

- El CSV trae los **nombres reales** (`VINTAGE`, `STRATEGY`, …), no los alias `dummiN`. El cargador (`src/data/load.py`) acepta los dos formatos, rechaza un encabezado que los mezcle e informa cuál encontró. El mapa queda en `data/contracts/column_map.json`.
- `MOB` es la columna llamada `MOB`, que el diccionario no le da alias. Si un archivo no la trae, la carga falla con un mensaje claro.
- El archivo tiene 1,089,833 filas, 22 columnas, 64,245 cohortes y ningún valor nulo. Todas sus columnas están en el diccionario.

## 2. Espacios al final de las categorías

- **Hallazgo:** el diccionario y los datos traen dos valores con espacio al final: `"Brands-NonPayroll-OffUs&LE "` y `"AAC Off Us Non-Payroll "`. Afecta a 194,376 filas de `STRATEGY_SEGMENT`.
- **Decisión:** al cargar se aplica `.str.strip()` a todas las dimensiones de texto, y `src/data/catalog.py` guarda los valores sin espacios. La comparación con el diccionario también se hace con strip.
- **Por qué:** con el espacio, el mismo segmento podría aparecer como dos categorías distintas, por ejemplo si un usuario lo escribe sin espacio en la app.

## 3. Catálogo frente a los datos

- `catalog.py` coincide con el diccionario en productos, canales, edades, segmentos y scores por familia, estrategia según vintage y rangos. Lo comprueban el script 01 y `tests/test_data_catalog.py`.
- **Hallazgo:** 5 de los 17 segmentos DARIA del diccionario no aparecen en los datos:
  - `Costco - Young <31`
  - `Costco - Non-LE`
  - `Costco - LE`
  - `Home Depot - Young <31`
  - `Home Depot - Non-LE`

  Se quedan en el catálogo porque son válidos según el contrato. Solo significa que no hay datos para ellos.

## 4. Validación del contrato

- Pasan todas las reglas bloqueantes y no hay avisos (`reports/01_datos/validacion.csv`). En particular:
  - No hay llaves repetidas.
  - No hay MOB mayor al posible para su vintage.
  - La estrategia siempre corresponde a su vintage.
  - `NCLAMTS = GCLAMTS − RECOVERYAMTS` se cumple en todas las filas; la diferencia máxima es del orden de 1e-10.
  - `EXPENSES_TOTAL` es exactamente la suma de sus tres componentes.
  - `EXPENSES_ACQ` vale 0 en todo MOB distinto de 0.
- **Regla agregada: `valor_nulo` (bloqueante).** No está en la lista de CLAUDE.md §5, pero ninguna columna del contrato admite vacíos y el rango del diccionario no se puede comprobar sobre un nulo. *Confirmar.*
- **Regla extendida: valores no enteros.** Para los conteos y la línea (`VINTAGE`, `MOB`, `ACCOUNTS`, `CREDITLINESAMTS`, `ENRACCTS`, `OVER30ACCTS`), un valor no entero cuenta como "fuera de rango", porque el diccionario dice "número entero".
- **Tolerancia de las igualdades:** |a − b| ≤ 1e-6 + 1e-9·|b|.

## 5. Columnas acumuladas (pregunta abierta 3)

La evidencia es cuántas veces sube, baja o se queda igual cada columna al pasar de un MOB al siguiente dentro de la cohorte. Detalle en `reports/01_datos/columnas_acumuladas.csv`.

| Columna | Baja | Sube | Conclusión |
|---|---|---|---|
| `OVER30ACCTS`, `OVER30AMTS` | 0% | 15.26% | Acumulada |
| `GCLAMTS`, `NCLAMTS`, `RECOVERYAMTS` | 0% | 100% | Acumulada |
| `REVENUE` | 0% | 87.99% | Acumulada (confirma lo dicho por Luis) |
| `ACCOUNTS`, `CREDITLINESAMTS`, `EXPENSES_ECM_FIX` | 0% | 0% | Constante en la cohorte |
| `ENRACCTS`, `ENRAMTS`, `EXPENSES_ECM_VAR`, `EXPENSES_TOTAL` | > 0% | > 0% | Del mes |
| `EXPENSES_ACQ` | — | — | Solo en MOB = 0 |

Consecuencia: `ACCOUNTS` es el número de cuentas colocadas y no cambia con el MOB, así que las cuentas se cuentan en MOB = 0.

## 6. Población de `REVENUE` y `EXPENSES_TOTAL`

- **Hallazgo:**
  - El diccionario describe ambas columnas como "de las cuentas que se fueron a Over30". Sin embargo, en las 356,290 filas con `OVER30ACCTS = 0`, el 49.78% tiene `REVENUE > 0` y el 100% tiene `EXPENSES_TOTAL > 0`. Como `OVER30ACCTS` es acumulado, esas filas no tienen ninguna cuenta en Over30.
  - Además, `EXPENSES_TOTAL` es exactamente `ACQ + VAR + FIX` en el 100% de las filas, y la descripción de esos tres componentes no los limita a Over30.
- **Decisión propuesta:** ambas columnas se refieren a **todas las cuentas de la cohorte**. La frase del diccionario parece un error de copia. No afecta S1 ni S3; importa si en la Fase 3 se usan ingresos o gastos para S4. *Confirmar.*

## 7. Denominador de S3 (pregunta abierta 4)

- En MOB = 6, `OVER30ACCTS > ENRACCTS` ocurre en **0 de 52,835 cohortes (0.00%)**, y tampoco en ningún otro MOB.
- **Decisión:** se mantiene `ENRACCTS` como denominador, tal como dice CLAUDE.md §6. No hace falta la propuesta de usar `ACCOUNTS`.
- **Referencia:**
  - La tasa ponderada a 6 meses es **0.0792 sobre `ENRACCTS`** y **0.0651 sobre `ACCOUNTS`**.
  - En MOB = 6, `ENRACCTS` es en promedio el 82% de `ACCOUNTS`. En MOB = 0 es solo el 32%, porque muchas cuentas todavía no tienen saldo.

## 8. Madurez y conjuntos de modelado

- **Madurez:** las vintages maduras a H = 6 son exactamente las 27 de 202401 a 202603. Las cohortes están completas: todas empiezan en MOB 0, no tienen huecos y llegan al MOB máximo posible.
- **`data/processed/s1_dataset.csv`:**
  - 64,245 cohortes en MOB = 0, de las 33 vintages, con 1,249,981 cuentas.
  - El objetivo es `LINEA_PROMEDIO = CREDITLINESAMTS / ACCOUNTS` y el peso es `ACCOUNTS`.
  - La línea promedio va de 8,800 a 480,000, con mediana de 38,318.
  - Incluye DARIA 2.0, que tiene línea aunque no tenga riesgo.
- **`data/processed/s3_dataset.csv`:**
  - 52,835 cohortes en MOB = 6, de las 27 vintages maduras.
  - El objetivo es `TASA_OVER30 = OVER30ACCTS / ENRACCTS` y el peso es `ENRACCTS`.
  - Por estrategia: 14.Recalibration 1.0 tiene 9,437 cohortes; 15.Recalibration 2.0, 27,616; r.Reca champ, 12,835; y DARIA 1.6, 2,947.
  - No incluye DARIA 2.0 y no se descartó ninguna fila por `ENRACCTS = 0`.
- **Observación para la Fase 2:**
  - Las cohortes son pequeñas: la mediana de `ENRACCTS` en S3 es 10 y el percentil 10 es 4.
  - El 35.1% de las cohortes de S3 tiene tasa 0, así que la etiqueta fraccional es ruidosa por fila.
  - Por eso se pondera por `ENRACCTS` y se evalúa con métricas ponderadas, no por fila.

## 9. Balance entre estrategias (ventana aleatoria 202511–202607)

- **Asignación:** la proporción observada de la retadora se desvía como máximo 2.2 puntos porcentuales de lo esperado (30% DARIA 1.6 y 50% DARIA 2.0). Detalle en `balance_asignacion_por_vintage.csv`.
- **Edad:** está balanceada; todas las \|SMD\| son menores o iguales a 0.06.
- **Producto y canal:** 5 categorías tienen |SMD| > 0.10. En total son 9 de las 32 comparadas, contando las 4 de score. Las causas son:
  - DARIA 1.6 **no tiene cuentas** de los productos Costco ni Beyond, ni del canal Costco.
  - DARIA 1.6 tiene más Clasica (34.4% contra 26.0%) y más Internet (28.4% contra 22.2%).
- **Score:** la "posición relativa" sale muy desbalanceada (SMD de la media +0.56), pero **no es comparable entre familias**. Cada familia tiene sus propios cortes (Recalibration usa 8 rangos de `<130` a `190+`; DARIA usa 9 de `≤154` a `205+`). Así que la diferencia refleja la escala, no necesariamente la población.
- **Estrategias evaluables en riesgo:** todas lo son menos **DARIA 2.0**, que solo tiene MOB 0 y 1 (`estrategias_evaluables.csv`).

## 10. Observaciones de la base sintética (no bloquean)

- **`GCLAMTS` ya es positivo en MOB = 1 en el 100% de las cohortes**, aunque solo el 2.42% tiene alguna cuenta en Over30 en ese mes. En datos reales la pérdida bruta llega después de la mora. Esto no afecta S1 ni S3; hay que tenerlo en cuenta si S4 usa pérdidas.
- `ACCOUNTS` va de 4 a 881 por fila: no hay cohortes de menos de 4 cuentas.

## 11. Puntos a decidir

1. Confirmar la regla `valor_nulo` como bloqueante (§4).
2. Confirmar que `REVENUE` y `EXPENSES_TOTAL` se refieren a todas las cuentas de la cohorte (§6).
3. Para S4 (Fase 3), elegir cómo comparar con DARIA 1.6:
   - restringir a los productos y canales que tienen las dos estrategias (soporte común, sin Costco ni Beyond), o
   - reportar esos perfiles como "sin evidencia".
4. Para la Fase 2, decidir si se mantiene la posición relativa del score como variable (está en CLAUDE.md §7). Como las escalas no son equivalentes entre familias, el modelo debería ver también la familia, que ya está prevista, para no mezclarlas.
