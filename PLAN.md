# PLAN.md — Plan de trabajo

> Cómo se usa: cada tarea es un notebook (o una pieza de `src/`). Al terminar una, marca su casilla `[x]`.
> No pases a la siguiente fase sin mi visto bueno. Las reglas y convenciones están en `CLAUDE.md`.

## Vista general

| Fase | Qué se hace | Estado |
|---|---|---|
| 0. Preparación | Repo, entorno Python 3.10, configuración | ✅ |
| 1. Datos | Cargar, validar y entender el dataset | ⬜ |
| 2. Modelos S1 y S3 | Líneas base y XGBoost, con métricas y versionado | ⬜ |
| — | **Fin del primer corte** | |
| 3. Estrategia y explicabilidad | S4 y SHAP | ⬜ (después) |
| 4. App y API | FastAPI + Streamlit | ⬜ (después) |
| 5. Asistente | RAG + Ollama + barreras | ⬜ (después) |
| 6. Cierre | Pruebas de extremo a extremo, documentación, empaquetado | ⬜ (después) |

**Cierre de cada notebook** (aplica a todos):
- [ ] Empieza con objetivo, entradas, salidas y criterio de terminado.
- [ ] Cada celda de código tiene su explicación (primero qué hace, luego por qué).
- [ ] Las gráficas están en celdas aparte.
- [ ] Las funciones reutilizables ya están en `src/` y tienen prueba en `tests/`.
- [ ] *Restart & Run All* sin errores.
- [ ] Esta casilla marcada en `PLAN.md`.

---

## Fase 0 — Preparación

**Alcance de la Fase 0:** solo infraestructura. No cargues datos ni crees notebooks todavía. Haz las tareas en este orden (el `.gitignore` va antes de cualquier commit).

- [x] **Localizar** el CSV de datos y el CSV del diccionario que ya están en el repo (dime dónde los encontraste). Si ya estaban en seguimiento de git, sácalos con `git rm --cached` antes del primer commit.
- [x] `.gitignore`: `data/raw/`, `data/processed/`, `models/` (binarios; los `.json` de metadatos sí se suben), `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`.
- [x] Crear la estructura de carpetas de `CLAUDE.md` §7 (con `.gitkeep` donde estén vacías).
- [x] **Mover** el CSV de datos a `data/raw/` y el diccionario a `data/contracts/diccionario_variables.csv`. Después de mover, `git status` no debe mostrar el CSV de datos como archivo nuevo ni rastreado.
- [x] Poner la propuesta en `docs/Propuesta_Tecnica.pdf` (si no está ya).
- [x] **Revisar la versión de Python** con `python --version`. Si no es 3.10, dime cuál es antes de crear el entorno: se puede desarrollar en una versión más nueva, pero el código debe seguir siendo compatible con 3.10. En Windows y macOS/Linux la activación del entorno cambia; ajusta los comandos a mi sistema.
- [x] `requirements/base.txt` y `requirements/ml.txt` (ver `CLAUDE.md` §3). Crear el entorno virtual, instalar y guardar las versiones resueltas en `requirements/lock.txt`, con la versión de Python anotada en la primera línea.
- [x] `configs/config.json`: semilla 42, horizonte H = 6, cortes de partición, rutas de datos y modelos.
- [x] `src/utils/`: lector de configuración y función `set_seed()`.
- [x] `README.md`: cómo crear el entorno, instalar y abrir los notebooks (máximo 5 pasos).
- [x] Primer commit de la fase, con `git status` revisado y mi confirmación del mensaje. **No hagas push.**

**Listo cuando:** `python --version` marca 3.10, `import pandas, numpy, sklearn, xgboost, shap, optuna, plotly` funciona, y alguien puede seguir el README desde cero.

---

## Fase 1 — Datos (`notebooks/01_datos/`)

Objetivo: tener cohortes limpias, validadas y entendidas, con el contrato de datos aprobado.

### 01 — `01_carga_y_diccionario.ipynb`
- [ ] Cargar el CSV de `data/raw/` en UTF-8; mostrar forma, tipos, nulos y memoria.
- [ ] Resolver los nombres de columna (`dummiN` vs reales) y estandarizar.
- [ ] Leer el diccionario y construir `src/data/catalog.py` con los catálogos (productos, canales, edades, segmentos y scores por familia, estrategia según vintage) y los rangos mínimos y máximos.
- [ ] Revisar espacios al final en categorías y decidir si se normalizan.
- **Sale:** DataFrame con nombres estándar; `catalog.py` con prueba.

### 02 — `02_validacion_contrato.ipynb`
- [ ] Implementar `src/data/validation.py` con pandas puro: reglas bloqueantes y avisos de `CLAUDE.md` §4.
- [ ] El validador devuelve un reporte (tabla) con: regla, fila, qué falló y qué hacer.
- [ ] Probar el validador con casos buenos y con casos rotos a propósito (en `tests/`).
- [ ] **Responder las preguntas 2 y 3 de `CLAUDE.md` §10** con evidencia de los datos (¿acumulado o estado? ¿`REVENUE` de todas las cuentas?).
- **Sale:** reporte de validación del dataset completo; decisiones documentadas.

### 03 — `03_exploracion_cohortes.ipynb`
- [ ] Cuentas por vintage, estrategia y segmento; participación de cada producto y canal.
- [ ] Distribución de score y edad por familia de estrategia.
- [ ] Cuántas cohortes y cuántas filas hay por MOB.
- **Sale:** tablas y gráficas descriptivas (gráficas en celdas aparte).

### 04 — `04_madurez_y_curvas.ipynb`
- [ ] Función que marca una cohorte como madura a H = 6 (`src/data/cohorts.py`); comprobar que son 27 vintages (202401–202603).
- [ ] Curvas de la tasa de Over30 por MOB para cada vintage y estrategia.
- [ ] Construir los dos conjuntos de modelado y guardarlos en `data/processed/` (CSV):
  - S1: `CREDITLINESAMTS / ACCOUNTS` en MOB = 0, todas las vintages.
  - S3: `OVER30ACCTS / ENRACCTS` en MOB = 6, solo cohortes maduras.
- **Sale:** `s1_dataset.csv` y `s3_dataset.csv`.

### 05 — `05_balance_entre_estrategias.ipynb`
- [ ] Aislar la ventana con asignación aleatoria (202511–202607) y comparar la mezcla de perfiles entre r.Reca champ y DARIA 1.6 (edad, producto, canal, posición de score).
- [ ] Marcar DARIA 2.0 como *no evaluable en riesgo*.
- **Sale:** diagnóstico de balance (lo usará S4 en la fase 3).

**Listo cuando:** el validador bloquea los archivos inválidos con errores legibles, y los dos conjuntos de modelado están generados y revisados por mí.

---

## Fase 2 — Modelos S1 y S3 (`notebooks/02_modelos_prediccion/`)

Objetivo: S1 y S3 con línea base y XGBoost, comparados en validación y prueba, versionados.

### 00 — `00_variables_y_particion.ipynb`
- [ ] `src/features/engineering.py`: one-hot (estrategia, segmento, producto, canal), edad ordinal, **posición relativa del score 0–1 dentro de su familia** + familia, mes de booking derivado de la vintage.
- [ ] Partición temporal por vintage (entrenamiento 202401–202508, validación 202509–202511, prueba 202512–202603; S1 hasta 202609).
- [ ] Prueba automática de que no hay fuga: ninguna vintage en dos conjuntos; ninguna columna de resultado, MOB, vintage numérica ni pesos entre las variables.
- **Sale:** matrices listas para S1 y S3.

### 01 — `01_s1_linea_base.ipynb`
- [ ] Media ponderada por producto y score.
- [ ] GLM Gamma (scikit-learn), con peso por cuentas.
- [ ] Métricas en validación y prueba: MAE, MAPE, R² y sesgo, ponderados.

### 02 — `02_s1_xgboost.ipynb`
- [ ] XGBoost sobre el logaritmo de la línea, peso = `ACCOUNTS`, árboles poco profundos.
- [ ] Optuna (50 pruebas), parada temprana (50 rondas), semilla 42, sobre validación.
- [ ] Mismas métricas, comparadas contra la línea base; reporte por estrategia.
- [ ] **Publicar solo si el MAE de prueba es al menos 10% menor que el de la línea base simple.**
- [ ] Guardar modelo y JSON de metadatos (`src/models/registry.py`).

### 03 — `03_s2_monto_total.ipynb`
- [ ] Función `S2 = cuentas planeadas × S1` en `src/`, con prueba (resultado exacto).

### 04 — `04_s3_linea_base.ipynb`
- [ ] Tasa ponderada por score y producto.
- [ ] GLM binomial (cada fila aporta tantos eventos como cuentas en Over30 y tantos no eventos como cuentas expuestas restantes).
- [ ] Métricas: AUC, KS, Gini, log loss y Brier ponderados; calibración por deciles.

### 05 — `05_s3_xgboost.ipynb`
- [ ] XGBoost con etiqueta fraccional, peso = `ENRACCTS`, restricción monótona decreciente en el score.
- [ ] Optuna y parada temprana como en S1; calibración ajustada **solo en validación**.
- [ ] Métricas contra la línea base; calibración por deciles; reporte por estrategia y por rango de edad.
- [ ] **Publicar solo si:** no es peor que el GLM, la razón observado/predicho está entre 0.95 y 1.05 y la tasa observada crece con el nivel de riesgo.
- [ ] Guardar modelo y JSON de metadatos.

### 06 — `06_resumen_modelos.ipynb`
- [ ] Tabla resumen de S1 y S3 (modelo vs línea base, validación vs prueba) y ficha corta de cada modelo para `docs/`.
- [ ] Recordatorio: el conjunto de prueba se evaluó **una sola vez**.

**Listo cuando:** ambos XGBoost superan a su línea base según los criterios, están versionados con sus metadatos y yo reviso las fichas.

---

## Fases siguientes (no empezar; se detallan al llegar)

### Fase 3 — Estrategia y explicabilidad
- `01_s4_comparador.ipynb`: solo la ventana de asignación aleatoria; dos métodos (diferencia observada con bootstrap sobre cohortes y predicción del mismo perfil bajo cada estrategia); se sugiere solo si el intervalo excluye cero y ambos coinciden en el signo.
- `03_explicabilidad/`: SHAP global, por segmento y por perfil (TreeExplainer; sumar one-hot por variable original); verificar que valor base + contribuciones = salida del modelo (tolerancia 0.0001).
- Módulo de reglas de consistencia (recorte de tasas a 0–1, dominio de entrenamiento, confianza por soporte histórico, nivel de riesgo por percentiles 50/80/95).

### Fase 4 — App y API
- FastAPI + Pydantic (local) y Streamlit; carga de CSV con validación; exportación; vista simple y avanzada. Empezar con una API simulada que devuelva el formato final.

### Fase 5 — Asistente
- Catálogo de consultas C1–C7, clasificador de intención, evidencia determinista, base documental, barreras numéricas y evaluación (≥ 60 preguntas). Aquí se decide si el asistente se divide en varios agentes.

### Fase 6 — Cierre
- Pruebas de extremo a extremo, manual de usuario, guía de instalación. Docker solo si se confirma que hace falta.
