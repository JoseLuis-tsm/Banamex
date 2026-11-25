# CLAUDE.md — Sistema de Inteligencia Crediticia (Tarjetas de Crédito)

> Contexto permanente para Claude Code. Léelo completo antes de tocar el repo.
> - Plan de trabajo: `PLAN.md` (ahí se marca el avance).
> - Propuesta original: `docs/Propuesta_Tecnica.pdf`.
> - Diccionario de variables: `data/contracts/diccionario_variables.csv`.
>
> Si algo de este archivo contradice la propuesta, avísame antes de decidir por tu cuenta.

---

## 1. Qué estamos construyendo

Una plataforma **local** que, para un perfil de adquisición de tarjetas de crédito, estima cuatro salidas:

| Salida | Pregunta | Tipo |
|---|---|---|
| S1. Línea promedio esperada | ¿Qué límite promedio recibe una cuenta de este perfil? | Regresión (XGBoost) |
| S2. Monto total esperado | ¿Cuánto crédito implica colocar N cuentas? | Cálculo: cuentas planeadas × S1 (no se predice) |
| S3. Tasa de impago esperada | ¿Qué fracción de cuentas estará en Over30 a 6 meses? | Tasa calibrada (XGBoost) |
| S4. Estrategia sugerida | ¿Qué estrategia rinde mejor para este perfil? | Comparación de escenarios con intervalo |

Cada resultado se explica con SHAP y se consulta en lenguaje natural con un asistente (RAG + Ollama) que **solo redacta sobre cifras ya calculadas**.

Los modelos se entrenan **una sola vez, fuera de línea**, con las vintages 202401 a 202609. La interfaz solo hace inferencia.

**Lo que NO hace:** no aprueba ni rechaza créditos (un riesgo crítico solo genera una alerta de revisión), no predice por cliente individual, no entrena modelos desde la interfaz y no afirma causalidad.

**Usuario final:** personal del banco sin especialización en riesgo de crédito ni ciencia de datos. Todo lo que vea debe ser claro y en español.

---

## 2. Estado y alcance actual

**Primer corte = Fases 0, 1 y 2 de `PLAN.md`:** repo, datos validados, y modelos S1 y S3 contra sus líneas base.

No empieces S4, SHAP, API, app, asistente ni Docker hasta que yo lo diga explícitamente. Empezamos de a poco, con cosas sólidas y sencillas, y escalamos después.

---

## 3. Reglas duras

1. **Python 3.10 con pip, archivos en UTF-8.** Abre y guarda siempre con `encoding="utf-8"`. No uses nada que exija 3.11 o superior (por ejemplo `tomllib` o `typing.Self`). Si en mi máquina hay una versión más nueva, se puede desarrollar en ella, pero el código debe seguir siendo compatible con 3.10; avísame qué versión hay.
2. **Solo las librerías del plan**, instaladas por capas desde `requirements/`:
   - `base.txt`: pandas, numpy, scikit-learn, plotly (+ herramientas mínimas de desarrollo: ipykernel, nbformat, pytest).
   - `ml.txt`: lo anterior + xgboost, shap, optuna.
   - `app.txt` y `assistant.txt` (después): streamlit, fastapi, pydantic, chromadb, ollama.
   - **Pandera no se usa por ahora.** Las validaciones se escriben con pandas puro. Si algún día se agrega, es opcional.
   - No agregues ninguna otra librería sin preguntarme.
3. **Todo local.** Sin APIs ni servicios externos de terceros, y ningún dato sale de la máquina. El modelo de lenguaje será Llama 3.1 8B con Ollama local. (La API interna con FastAPI, en la fase 4, corre en localhost.)
4. **Docker es una posibilidad, no un supuesto.** Nada puede depender de Docker: todo debe correr con `python -m venv` + `pip install -r requirements/...`.
5. **Semilla 42** en todo lo aleatorio (numpy, xgboost, optuna, bootstrap, particiones).
6. **Datos fuera de git:** `data/raw/` y `data/processed/` van en `.gitignore`. Nunca subas el CSV. Los modelos entrenados (binarios) tampoco; sus metadatos en JSON sí.
7. **Formatos sin dependencias extra:** tablas intermedias en **CSV** (no parquet, porque `pyarrow` no está en el plan) y parámetros en **JSON** (`configs/config.json`; no YAML, porque `pyyaml` no está en el plan).
8. **Sin fuga de información:**
   - La partición es **temporal por vintage**, nunca aleatoria por fila.
   - Codificadores, percentiles y calibración se ajustan sin ver el conjunto de prueba.
   - El conjunto de prueba se usa **una sola vez**, al final.
9. **Las columnas de resultado nunca son variables de entrada** (ver §4).
10. **Ninguna cifra inventada.** Todo número que se reporte sale de un cálculo ejecutado en el repo.
11. **No generes datos sintéticos** por tu cuenta. Si falta el CSV de datos, avísame y espera.

---

## 4. Datos (resumen del contrato)

**Qué es una fila:** el resumen de todas las cuentas que comparten `VINTAGE`, `STRATEGY`, `STRATEGY_SEGMENT`, `PRODUCT`, `CHANNEL`, `AGE_TAG` y `SCORE_TAG`, observadas en un mes de desempeño `MOB`. Esas **8 columnas son la llave única**; no deben repetirse.

**Corte estático a septiembre de 2026 (202609).** El MOB máximo de una vintage son los meses entre ella y 202609 (202401 → 32, 202402 → 31, …, 202608 → 1, 202609 → 0).

**Grupos de variables:**

| Grupo | Variables | Rol |
|---|---|---|
| Dimensiones | VINTAGE, STRATEGY, STRATEGY_SEGMENT, PRODUCT, CHANNEL, AGE_TAG, SCORE_TAG, MOB | Atributos de la cohorte |
| Volumen y exposición | ACCOUNTS, ENRACCTS, CREDITLINESAMTS, ENRAMTS | Pesos, denominadores y objetivos |
| Resultados | OVER30ACCTS, OVER30AMTS, GCLAMTS, NCLAMTS, RECOVERYAMTS, REVENUE, EXPENSES_ACQ, EXPENSES_ECM_VAR, EXPENSES_ECM_FIX, EXPENSES_TOTAL | Solo construyen objetivos; **nunca entradas** |

**Nombres de columna:** el diccionario trae dos nombres por variable (`dummiN` y el nombre real, por ejemplo `dummi1` = `VINTAGE`). Verifica con el CSV de datos cuál se usa y **estandariza a los nombres reales** al cargar (dejando la equivalencia documentada).

**Estrategia según vintage:**

| Vintages | STRATEGY | Asignación |
|---|---|---|
| 202401–202405 | 14.Recalibration 1.0 | Determinista por fecha |
| 202406–202507 | 15.Recalibration 2.0 | Determinista por fecha |
| 202508–202510 | r.Reca champ | Determinista por fecha |
| 202511–202607 | r.Reca champ (70%) / DARIA 1.6 (30%) | Aleatoria |
| 202608–202609 | r.Reca champ (50%) / DARIA 2.0 (50%) | Aleatoria |

**Dos familias de estrategia:** *Recalibration* (7 segmentos, 8 rangos de score) y *DARIA* (17 segmentos, 9 rangos de score). PRODUCT, CHANNEL y AGE_TAG son compartidos; las comparaciones entre estrategias se hacen sobre ellos.

**Madurez:** el horizonte es **H = 6 meses**. Una cohorte es madura si ya se observó hasta MOB = 6. Vintages maduras: **202401 a 202603 (27)**.

**DARIA 2.0** tiene MOB 0 y 1 solamente: tiene línea asignada pero no desempeño de riesgo, así que se marca como *no evaluable en riesgo*.

**Reglas de validación del contrato** (el validador debe devolver un reporte legible: qué falló, en qué fila y qué hacer):

- *Bloquean la carga:* llave repetida; columna faltante; categoría fuera de catálogo; estrategia que no corresponde a su vintage; MOB mayor al posible para su vintage; valor fuera del rango del diccionario; `NCLAMTS ≠ GCLAMTS − RECOVERYAMTS`; `EXPENSES_TOTAL ≠ EXPENSES_ACQ + EXPENSES_ECM_VAR + EXPENSES_ECM_FIX`; `EXPENSES_ACQ ≠ 0` con MOB ≠ 0. (Las igualdades se comparan con una tolerancia numérica pequeña.)
- *Solo avisan:* cuentas en Over30 mayores a las expuestas (`OVER30ACCTS > ENRACCTS`); columnas no declaradas en el diccionario (se descartan).

**Detalle a revisar en el diccionario:** algunos valores de catálogo traen un espacio al final (por ejemplo `"Brands-NonPayroll-OffUs&LE "`, `"AAC Off Us Non-Payroll "`). Compara contra los datos y normaliza con `.str.strip()` si procede, documentando la decisión.

---

## 5. Definición de las salidas

- **S1:** objetivo = `CREDITLINESAMTS / ACCOUNTS` en **MOB = 0** (la línea que se asigna al booking). XGBoost sobre el **logaritmo** de la línea, con peso = `ACCOUNTS`. Líneas base: media ponderada por producto y score; GLM Gamma.
- **S2:** `S2 = cuentas planeadas × S1`. Función determinista, no modelo.
- **S3:** objetivo = `OVER30ACCTS / ENRACCTS` en **MOB = H = 6**, como etiqueta fraccional con peso = `ENRACCTS`, efecto **decreciente** del score (restricción monótona) y calibración para que la tasa predicha coincida con la observada. Líneas base: tasa ponderada por score y producto; GLM binomial. *Provisional hasta resolver la pregunta 2 de §9.*
- **S4:** (fase 3, todavía no) comparación de escenarios; se llama *sugerida*, no *óptima*.

---

## 6. Variables de entrada, partición y criterios

**Entran:** STRATEGY, STRATEGY_SEGMENT, PRODUCT y CHANNEL en one-hot; AGE_TAG como ordinal (1 a 3); el score como **posición relativa entre 0 y 1 dentro de su familia** más la familia (Recalibration o DARIA); y el **mes de booking (1–12)** derivado de la vintage, para capturar estacionalidad.

**No entran:** la vintage como número, el MOB, ACCOUNTS y ENRACCTS (son pesos) ni ninguna columna de resultado.

**Partición temporal por vintage:**

| Conjunto | Vintages | Uso |
|---|---|---|
| Entrenamiento | 202401–202508 (20) | Ajustar el modelo |
| Validación | 202509–202511 (3) | Hiperparámetros y calibración |
| Prueba | 202512–202603 (4); para S1, hasta 202609 | Evaluación final, una sola vez |

DARIA aparece solo en validación y prueba, así que esa parte mide extrapolación: **las métricas se reportan por estrategia**.

**Configuración común de XGBoost:** Optuna con 50 pruebas, parada temprana de 50 rondas, semilla 42 y árboles poco profundos (hay pocas vintages). **Un XGBoost se publica solo si supera a su línea base.**

| Salida | Métricas | Criterio en prueba |
|---|---|---|
| S1 | MAE, MAPE, R² y sesgo, ponderados | MAE al menos 10% menor que el de la línea base simple |
| S3 | AUC, KS, Gini, log loss y Brier ponderados; calibración por deciles | No peor que el GLM; razón observado/predicho entre 0.95 y 1.05; tasa observada creciente por nivel de riesgo |

Para AUC y calibración, cada fila aporta tantos eventos como cuentas en Over30 y tantos no eventos como cuentas expuestas restantes. S3 también se reporta por rango de edad.

**Versionado:** cada modelo se guarda con un JSON de metadatos: fecha, vintages usadas, variables, hiperparámetros, métricas, **versión de datos (hash SHA-256 del CSV)**, umbrales de riesgo y versiones de librerías.

---

## 7. Estructura del repo

```
├── CLAUDE.md  PLAN.md  README.md
├── configs/config.json            # semilla, H, cortes de partición, rutas
├── requirements/                  # base.txt, ml.txt (después app.txt, assistant.txt)
├── data/
│   ├── raw/                       # CSV original (NO va a git)
│   ├── processed/                 # tablas derivadas en CSV (NO va a git)
│   └── contracts/                 # diccionario de variables y reglas de validación
├── notebooks/
│   ├── 01_datos/
│   ├── 02_modelos_prediccion/
│   ├── 03_explicabilidad/         # (después) SHAP
│   ├── 04_asistente_rag/          # (después) RAG, clasificador, barreras
│   └── 05_evaluacion/             # (después)
├── src/                           # funciones reutilizables
│   ├── utils/  data/  features/  models/  explain/  assistant/  consistency/
├── models/                        # modelos versionados + metadatos
├── api/  app/                     # (después) FastAPI y Streamlit
├── docs/                          # propuesta, fichas de modelo, manual
└── tests/
```

**Regla clave: los notebooks explican y ejecutan; la lógica reutilizable vive en `src/`.** Un notebook define o ajusta una función, la prueba, la documenta y deja la versión final en `src/` para importarla. Así, cuando llegue la API y el frontend, no hay que copiar código entre archivos.

**Un notebook = una tarea.** Por ejemplo: un notebook para el XGBoost de S1, otro para la línea base de S1, otro para el XGBoost de S3, etc.

---

## 8. Convenciones

### Notebooks

Cada notebook empieza con una celda de texto que dice: **objetivo**, **qué entra** (archivos o funciones), **qué sale** y **cómo sé que terminó bien**. Después:

- Cada celda de código va precedida de una celda de texto breve.
- **Al explicar código, primero qué hace** (qué variable guarda qué, con qué función y cómo) **y después el porqué** (el fundamento).
- Cuando hay una fórmula, un bloque que la **traduce a código**, símbolo por símbolo: qué variable del código representa cada término. Asume que domino la matemática; lo que necesito ver es cómo se mapea a las líneas.
- Las **gráficas van en una celda aparte** de la que calcula los datos. Plotly para gráficas; matplotlib solo para las gráficas nativas de SHAP.
- Antes de dar un notebook por terminado: *Restart & Run All* sin errores, con salidas guardadas.
- Semilla fija y rutas leídas de `configs/config.json`, nunca escritas a mano.

### Código en `src/`

- Funciones pequeñas, con nombre claro, sin estado global.
- Nombres de funciones y variables en inglés (`snake_case`); texto, comentarios y docstrings en **español**.
- Docstring corto: qué hace, parámetros, qué devuelve.
- Type hints básicos.
- Cada función que se use en más de un notebook tiene una prueba en `tests/` (con `pytest`).

---

## 9. Cómo quiero que trabajes conmigo

- Avanza **un notebook a la vez** y dime qué hiciste y qué sigue antes de pasar al siguiente. No saltes de fase sin mi visto bueno.
- Al terminar una tarea, marca su casilla en `PLAN.md`.
- Si modificas un notebook que ya existe, **cambia solo las celdas necesarias** y dime cuáles y dónde; no lo reescribas completo.
- Si encuentras algo ambiguo en los datos o en la propuesta, no lo resuelvas en silencio: documéntalo en el notebook, propón una decisión y pregúntame.
- Si una librería no instala en Python 3.10, dímelo antes de buscar alternativas.

### Git

- Un commit por tarea terminada (por ejemplo, un notebook), con mensaje corto en español y en imperativo: `Agrega notebook de validación del contrato`.
- Antes de cada commit, revisa `git status` y confirma que no entran `data/`, modelos binarios, `.venv/` ni `.ipynb_checkpoints/`. No uses `git add .` a ciegas.
- **Pídeme confirmación antes de** `git push`, `git reset --hard`, `--force`, borrar ramas o cualquier comando que reescriba historial. Por ahora los `push` los hago yo.
- Trabaja en una rama por fase (`fase-1-datos`, `fase-2-modelos`…), no directo en `main`. Antes de empezar una sesión, haz `git pull`.
- Mi compañero y yo trabajamos **notebooks distintos** para no chocar: los `.ipynb` son JSON y sus conflictos de merge son difíciles de resolver. Si dos personas necesitan tocar el mismo notebook, avísame antes.

---

## 10. Preguntas abiertas (resolverlas en la Fase 1, con los datos)

1. **¿Dónde está el CSV de datos?** Debe quedar en `data/raw/`. Si no está, avísame y no sigas.
2. **¿`OVER30ACCTS` es el estado en ese MOB o el acumulado de las cuentas que alguna vez llegaron?** Compruébalo con los datos: dentro de una misma cohorte, si `OVER30ACCTS` nunca baja al subir el MOB, es un acumulado; si sube y baja, es el estado de cada mes. Documenta el resultado en el notebook `02_validacion_contrato`; define cómo se construye S3.
3. **¿`REVENUE` y `EXPENSES_TOTAL` son de todas las cuentas del grupo o solo de las de Over30?** Sus descripciones dicen "Over30", pero las de sus componentes de gasto no. Revisa que `EXPENSES_TOTAL = ACQ + VAR + FIX` se cumpla en los datos.
4. **¿Qué nombres usan las columnas del CSV, los `dummiN` o los reales?** (ver §4).
5. **Fecha límite y reparto del equipo:** pendientes (los define Luis; no los inventes).

---

## 11. Glosario mínimo

- **Booking:** momento en que se abre la cuenta. **Vintage:** mes y año del booking (`AAAAMM`).
- **MOB (month on books):** mes de desempeño relativo al booking; MOB 0 es el mes del booking.
- **Cohorte / perfil:** conjunto de cuentas con las mismas 7 dimensiones.
- **Over30:** cuentas que se fueron a mora de más de 30 días (definición habitual; confirmar la exacta con los datos).
- **GCL / NCL:** pérdida bruta / pérdida neta (`NCL = GCL − recuperación`).
- **Cuentas expuestas (`ENRACCTS`):** cuentas con saldo; son el denominador de la tasa de impago.
- **Reca champ / Recalibration / DARIA:** familias de estrategia de adquisición (ver §4).
