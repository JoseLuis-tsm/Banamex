# Ficha del modelo S3 — Tasa de impago esperada

> Generada por `scripts/02_modelos_prediccion/06_resumen_modelos.py` a partir de `metadata.json` y de los reportes de métricas. No editar a mano: se regenera al correr el script.

- **Estado:** **Publicado** (models/s3/actual.json).
- **Versión:** `s3-e212982330` · **Datos (SHA-256):** `7f52a02f2a0b81de…`
- **Base de datos sintética:** sí. Las cifras describen la base sintética, no datos reales.
- **Partición temporal por vintage:** entrenamiento 202401–202508 (20), validación 202509–202511 (3), prueba 202512–202603 (4). El modelo final se reentrena con 27 vintages.
- **Uso de la prueba:** el conjunto de prueba se evaluó **una sola vez**, al final, con el modelo entrenado solo con entrenamiento; hiperparámetros, retransformación y calibración se eligieron en validación.
- **Variables:** 57 (one-hot de estrategia, segmento, producto y canal; edad ordinal; posición del score 0–1 en su familia; familia; mes de booking). No entran la vintage, el MOB, los pesos ni las columnas de resultado.
- **Hiperparámetros:** profundidad 2, eta 0.0232, 742 rondas; Optuna 50 pruebas, parada temprana 50, semilla 42.

## Qué estima

La tasa acumulada de Over30 a 6 meses de un perfil: `OVER30ACCTS / ENRACCTS` en MOB = 6 (`OVER30ACCTS` es acumulado). XGBoost con etiqueta fraccional, peso = `ENRACCTS` y restricción monótona decreciente en la posición del score (violaciones en prueba: 0). Calibración **Platt** ajustada en validación: a = 1.0006, b = 0.0047. Dos parámetros (validación de 3 vintages), estrictamente creciente (conserva orden, AUC y monotonía) y sin escalones.

## Resultados (totales)

| modelo | conjunto | auc | ks | gini | log_loss | brier | obs_pred |
|---|---|---|---|---|---|---|---|
| Base simple (score × producto) | validacion | 0.6232 | 0.1854 | 0.2464 | 0.27343 | 0.0731 | 1.051 |
| Base simple (score × producto) | prueba | 0.6083 | 0.1761 | 0.2166 | 0.29579 | 0.0805 | 1.155 |
| GLM binomial | validacion | 0.6384 | 0.1970 | 0.2767 | 0.27130 | 0.0727 | 0.976 |
| GLM binomial | prueba | 0.6454 | 0.2146 | 0.2908 | 0.29002 | 0.0794 | 0.940 |
| XGBoost calibrado | validacion | 0.6392 | 0.1994 | 0.2785 | 0.27120 | 0.0727 | 1.000 |
| XGBoost calibrado | prueba | 0.6462 | 0.2117 | 0.2925 | 0.28976 | 0.0794 | 1.012 |

## Prueba por estrategia

| grupo | auc | log_loss | obs_pred | peso |
|---|---|---|---|---|
| DARIA 1.6 | 0.6325 | 0.35221 | 1.048 | 40,263 |
| r.Reca champ | 0.6309 | 0.26282 | 0.990 | 93,365 |

## Prueba por rango de edad

| grupo | auc | log_loss | obs_pred | peso |
|---|---|---|---|---|
| 01\| Young < 25 | 0.6484 | 0.28697 | 1.018 | 29,799 |
| 02\| Young 25-30 | 0.6471 | 0.29240 | 1.013 | 51,957 |
| 03\| Mature >30 | 0.6440 | 0.28871 | 1.008 | 51,872 |

## Niveles de riesgo (prueba)

Umbrales: P50 = 0.0647, P80 = 0.1023, P95 = 0.1482. Percentiles 50, 80 y 95 de la tasa predicha calibrada en entrenamiento, ponderados por ENRACCTS. Bajo ≤ P50 < Medio ≤ P80 < Alto ≤ P95 < Crítico.

| nivel | cohortes | cuentas_expuestas | tasa_observada | tasa_predicha |
|---|---|---|---|---|
| Bajo | 3,067 | 51,897 | 5.09% | 5.04% |
| Medio | 2,216 | 37,878 | 8.18% | 7.94% |
| Alto | 1,357 | 23,594 | 11.54% | 11.40% |
| Crítico | 1,185 | 20,259 | 17.07% | 17.10% |

## Criterio de publicación

En prueba: log loss ≤ y AUC ≥ los del GLM; obs/pred entre 0.95 y 1.05; tasa observada creciente por nivel de riesgo; sin violaciones de la monotonía en el score.

- ✔ log loss no peor que glm
- ✔ auc no peor que glm
- ✔ obs pred en rango
- ✔ tasa crece con nivel
- ✔ monotonia en score

Resultado: **aprobado**.

## Limitaciones

- DARIA 1.6 solo aparece en validación y prueba: sus métricas miden extrapolación. DARIA 2.0 no es evaluable en riesgo (solo MOB 0 y 1).
- La mejora sobre el GLM es pequeña; el poder discriminante (AUC) es moderado en todos los modelos de la base sintética.
- Los umbrales de riesgo se calculan aquí para el criterio de publicación; la Fase 3 los usa en las reglas de consistencia.
- La herramienta no aprueba ni rechaza créditos: un riesgo Crítico solo genera una alerta de revisión. No afirma causalidad.
