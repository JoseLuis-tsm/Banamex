# Ficha del modelo S1 — Línea promedio esperada

> Generada por `scripts/02_modelos_prediccion/06_resumen_modelos.py` a partir de `metadata.json` y de los reportes de métricas. No editar a mano: se regenera al correr el script.

- **Estado:** **Publicado** (models/s1/actual.json).
- **Versión:** `s1-ec19747b9b` · **Datos (SHA-256):** `7f52a02f2a0b81de…`
- **Base de datos sintética:** sí. Las cifras describen la base sintética, no datos reales.
- **Partición temporal por vintage:** entrenamiento 202401–202508 (20), validación 202509–202511 (3), prueba 202512–202609 (10). El modelo final se reentrena con 33 vintages.
- **Uso de la prueba:** el conjunto de prueba se evaluó **una sola vez**, al final, con el modelo entrenado solo con entrenamiento; hiperparámetros, retransformación y calibración se eligieron en validación.
- **Variables:** producto y rango de score (tabla de medias ponderadas; si la combinación no se vio, solo producto y luego global).

## Qué estima

La línea de crédito promedio por cuenta que recibe un perfil al booking: `CREDITLINESAMTS / ACCOUNTS` en MOB = 0. **Modelo publicado: línea base simple**, la línea media ponderada por producto × rango de score (Σ `CREDITLINESAMTS` / Σ `ACCOUNTS` del grupo); si la combinación no se vio, usa solo el producto y, al final, el valor global. El modelo final se recalcula con todas las vintages.

## Resultados (totales)

| modelo | conjunto | mae | mape_pct | r2 | sesgo_pct |
|---|---|---|---|---|---|
| Base simple (producto × score) | validacion | 15,482 | 37.7% | 0.797 | -1.3% |
| Base simple (producto × score) | prueba | 15,443 | 37.0% | 0.758 | -0.9% |
| GLM Gamma | validacion | 15,459 | 37.0% | 0.796 | -3.1% |
| GLM Gamma | prueba | 15,442 | 36.0% | 0.754 | -3.9% |
| XGBoost | validacion | 15,640 | 37.6% | 0.786 | -2.5% |
| XGBoost | prueba | 15,892 | 36.6% | 0.713 | -4.7% |

## Prueba por estrategia

| modelo | grupo | mae | mape_pct | sesgo_pct | peso |
|---|---|---|---|---|---|
| Base simple (producto × score) | DARIA 1.6 | 14,540 | 37.6% | -1.9% | 89,468 |
| Base simple (producto × score) | DARIA 2.0 | 14,422 | 36.9% | -2.3% | 39,362 |
| Base simple (producto × score) | r.Reca champ | 15,925 | 36.9% | -0.4% | 250,943 |
| GLM Gamma | DARIA 1.6 | 14,780 | 36.8% | -6.0% | 89,468 |
| GLM Gamma | DARIA 2.0 | 14,762 | 36.0% | -6.6% | 39,362 |
| GLM Gamma | r.Reca champ | 15,785 | 35.8% | -2.9% | 250,943 |
| XGBoost | DARIA 1.6 | 15,853 | 37.1% | -10.9% | 89,468 |
| XGBoost | DARIA 2.0 | 16,431 | 37.2% | -12.4% | 39,362 |
| XGBoost | r.Reca champ | 15,821 | 36.4% | -1.7% | 250,943 |

## Criterio de publicación

Un XGBoost se publica solo si supera a su línea base (MAE de prueba al menos 10% menor). El XGBoost evaluado (`s1-4eeb609786`) obtuvo MAE de prueba 15,892, una mejora de **-2.9%** sobre la base simple → **no aprobado**. Línea base simple, por decisión de Luis en la Fase 2 (opción A).

## Limitaciones

- DARIA solo aparece en validación y prueba: sus métricas miden extrapolación (el modelo no vio DARIA al entrenar).
- Ningún modelo explica la variación de la línea entre cohortes de un mismo perfil; el error relativo (MAPE) es parecido en todos los modelos.
- La base simple solo distingue producto y rango de score: perfiles que difieren en estrategia, segmento, canal o edad reciben la misma línea. Los rangos de score DARIA no aparecen en entrenamiento: en la evaluación de prueba esos perfiles usaron el respaldo por producto; el modelo final, recalculado con todas las vintages, sí los incluye.
- La herramienta no aprueba ni rechaza créditos ni predice por cliente individual.
