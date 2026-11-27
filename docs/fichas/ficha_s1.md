# Ficha del modelo S1 — Línea promedio esperada

> Generada por `scripts/02_modelos_prediccion/06_resumen_modelos.py` a partir de `metadata.json` y de los reportes de métricas. No editar a mano: se regenera al correr el script.

- **Estado:** **No publicado**: no cumplió el criterio de publicación.
- **Versión:** `s1-4eeb609786` · **Datos (SHA-256):** `7f52a02f2a0b81de…`
- **Base de datos sintética:** sí. Las cifras describen la base sintética, no datos reales.
- **Partición temporal por vintage:** entrenamiento 202401–202508 (20), validación 202509–202511 (3), prueba 202512–202609 (10). El modelo final se reentrena con 33 vintages.
- **Uso de la prueba:** el conjunto de prueba se evaluó **una sola vez**, al final, con el modelo entrenado solo con entrenamiento; hiperparámetros, retransformación y calibración se eligieron en validación.
- **Variables:** 57 (one-hot de estrategia, segmento, producto y canal; edad ordinal; posición del score 0–1 en su familia; familia; mes de booking). No entran la vintage, el MOB, los pesos ni las columnas de resultado.
- **Hiperparámetros:** profundidad 2, eta 0.0244, 918 rondas; Optuna 50 pruebas, parada temprana 50, semilla 42.

## Qué estima

La línea de crédito promedio por cuenta que recibe un perfil al booking: `CREDITLINESAMTS / ACCOUNTS` en MOB = 0. XGBoost sobre el logaritmo de la línea, con peso = `ACCOUNTS`. Retransformación elegida en validación: **duan** (factor del modelo final 1.0722).

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

MAE de prueba al menos 10% menor que el de la línea base simple. Mejora obtenida: **-2.9%** → **no aprobado**.

## Limitaciones

- DARIA solo aparece en validación y prueba: sus métricas miden extrapolación (el modelo no vio DARIA al entrenar).
- Ningún modelo explica la variación de la línea entre cohortes de un mismo perfil; el error relativo (MAPE) es parecido en todos los modelos.
- La herramienta no aprueba ni rechaza créditos ni predice por cliente individual.
