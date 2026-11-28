# Consistencia y S4: decisiones de la Fase 3

Decisiones aprobadas por Luis antes de implementar `scripts/03_estrategia_y_explicabilidad/03` a `05`. Las cifras salen de esos scripts (base sintética).

## 1. Perfil fuera del dominio

- **Definición:** un perfil está fuera del dominio de un modelo si su combinación **(familia, segmento, producto, canal)** no aparece en los datos con que se entrenó ese modelo. La edad y el score pueden variar libremente.
- **Qué pasa:** el perfil no recibe la salida de ese modelo (queda vacía) y sale con estado y motivo. Si está fuera del dominio de S1 no recibe línea ni monto; si solo está fuera del de S3, recibe línea pero no tasa.
- **Qué no es "fuera de dominio":** una combinación que contradice el catálogo (por ejemplo, un score DARIA con una estrategia Recalibration) da un error de validación. DARIA 2.0 sale siempre *no evaluable en riesgo*, porque solo tiene MOB 0 y 1.
- **Referencia de los modelos publicados:** todas sus vintages de entrenamiento. Con ella, 32 de las 19,235 filas de prueba de S1 quedan fuera del dominio de S3.
- **Diagnóstico:** se usa como referencia lo que vio el modelo de evaluación antes de la prueba (entrenamiento más validación). Con ella, el 4.4% de las filas de prueba de S3 queda fuera del dominio, todas de DARIA 1.6.

## 2. Perfiles similares y confianza

- **Definición:** perfiles similares son los que tienen la misma **familia, segmento, producto, canal y edad**, con cualquier score. Las estrategias de una misma familia se juntan.
- **Conteo:** se suman las cuentas (`ACCOUNTS`) de las cohortes maduras a 6 meses.
- **Niveles:** Alta con 1,000 cuentas o más, Media de 200 a 999 y Baja con menos de 200.
- **Distribución sobre los perfiles distintos de las cohortes maduras:** 65.6% Alta, 19.7% Media y 14.7% Baja.
- **DARIA 2.0: "No aplica"** (decisión de Luis). La confianza acompaña a la tasa de impago, y DARIA 2.0 no la tiene. Si se calculara, saldría de la historia madura de DARIA 1.6 en perfiles parecidos (Baja o Media, nunca Alta) y podría leerse como una confianza sobre una tasa que no se muestra. Las cuentas similares se conservan como dato informativo.
- **Distribución sobre las filas de prueba de S1:** 11,525 Alta, 2,375 Media, 3,552 Baja y 1,783 No aplica (DARIA 2.0). Sin contar DARIA 2.0, son 66.0% Alta, 13.6% Media y 20.4% Baja. Hay más Baja que en las cohortes maduras porque estas filas incluyen vintages recientes, con menos historia madura.

## 3. Umbrales de riesgo: desviación de la propuesta

- **Qué pide la propuesta:** percentiles 50, 80 y 95 de la **tasa histórica ponderada** (la observada).
- **Qué se usa:** los percentiles 50, 80 y 95 de la **tasa predicha calibrada** en entrenamiento, ponderados por `ENRACCTS`, guardados en el `metadata.json` de S3.

| Percentil | Valor |
|---|---|
| P50 | 0.0647 |
| P80 | 0.1023 |
| P95 | 0.1482 |

- **Por qué:** la tasa observada por cohorte es muy ruidosa. El 35% de las cohortes tiene tasa 0, así que sus percentiles reflejan el tamaño de las cohortes más que el riesgo.
- **Validación:** con los umbrales sobre la tasa predicha, la tasa observada en prueba crece por nivel: 5.1%, 8.2%, 11.5% y 17.1% (Fase 2).

## 4. S4, método 2: predicción fuera de muestra

- **Cómo se predice:** en cada perfil (producto × canal × edad) se predicen con S3 las cohortes reales de cada estrategia, cada una con sus propios segmentos y scores. Luego se promedian ponderando por `ENRACCTS`.
- **Por qué fuera de muestra:** el S3 publicado ya vio esas cohortes al entrenar. Por eso cada una de las 5 vintages de la ventana se predice con un S3 reentrenado sin ella, con los mismos hiperparámetros, rondas y calibración del publicado.
- **Concordancia de signo** en los perfiles concluyentes del método 1:
  - Fuera de muestra: 92 de 93 (98.9%).
  - Con el modelo publicado en muestra, solo como referencia: el mismo 98.9%.
  - La predicción por cohorte cambia poco al quitar una vintage: 0.02 pp de mediana y 1.3 pp como máximo. Son árboles de profundidad 2, con mucha regularización.
- **Intervalo:** el intervalo bootstrap del método 2 solo refleja la variación entre cohortes. La decisión usa el intervalo del método 1.
- **Resultado de la regla:**

| Resultado | Perfiles |
|---|---|
| Sugerida r.Reca champ | 91 |
| Sugerida DARIA 1.6 | 1 |
| Sin diferencia concluyente | 102 |
| Sin evidencia (fuera del soporte común) | 28 |
| DARIA 2.0 | No evaluable en riesgo |
