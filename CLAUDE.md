# CLAUDE.md — Sistema de Inteligencia Crediticia (Tarjetas de Crédito)

> Contexto permanente para Claude Code. Léelo completo al iniciar cada sesión.
> - Plan de trabajo y avance: `PLAN.md`
> - Guía de uso del repo (instalación, scripts, dónde está cada cosa): `README.md`
> - Propuesta original: `docs/Propuesta_Tecnica.pdf`
> - Diccionario de variables: `data/contracts/diccionario_variables.csv`
> - Contratos entre frentes: `docs/contratos/`
> - Decisiones por fase, con su evidencia: `docs/datos_fase1.md`, `docs/fichas/`, `docs/consistencia_fase3.md`
>
> Si algo de este archivo contradice la propuesta, avísame antes de decidir por tu cuenta.

---

## 1. Qué estamos construyendo

Una plataforma **local** que, para un perfil de adquisición de tarjetas de crédito, estima cuatro salidas:

| Salida | Pregunta | Tipo |
|---|---|---|
| S1. Línea promedio esperada | ¿Qué límite promedio recibe una cuenta de este perfil? | Regresión. **Publicada: línea base simple** (producto × score), porque el XGBoost no cumplió el criterio |
| S2. Monto total esperado | ¿Cuánto crédito implica colocar N cuentas? | Cálculo: cuentas planeadas × S1 (no se predice) |
| S3. Tasa de impago esperada | ¿Qué fracción de cuentas estará en Over30 a 6 meses? | Tasa calibrada (XGBoost, publicado) |
| S4. Estrategia sugerida | ¿Qué estrategia rinde mejor para este perfil? | Comparación de escenarios con intervalo |

Cada resultado se explica (SHAP en S3; descomposición exacta por factores en S1) y se consulta en lenguaje natural con un asistente (RAG + Ollama) que **solo redacta sobre cifras ya calculadas**.

Los modelos se entrenan **una sola vez, fuera de línea**, con las vintages 202401 a 202609. La interfaz solo hace inferencia.

**Lo que NO hace:** no aprueba ni rechaza créditos (un riesgo crítico solo genera una alerta de revisión), no predice por cliente individual, no entrena modelos desde la interfaz y no afirma causalidad.

**Usuario final:** personal del banco sin especialización en riesgo de crédito ni ciencia de datos. Todo lo que vea debe ser claro y en español. Los analistas de riesgo y la gerencia usan una vista avanzada y una ejecutiva de la misma herramienta.

---

## 2. Equipo, reparto y con quién hablas

Dos personas trabajan **en paralelo** en el mismo repo:

| Persona | Frente | Carpetas propias |
|---|---|---|
| **Luis** | Datos, modelos (S1, S2, S3, S4), consistencia, SHAP, evidencia determinista y RAG | `scripts/01_datos`, `02_modelos_prediccion`, `03_estrategia_y_explicabilidad`, `05a_evidencia_rag`; `src/data`, `features`, `models`, `strategy`, `consistency`, `explain`, `evidence`, `rag`; `data/contracts`, `mock`, `knowledge`; `models/` |
| **Ernesto** | El agente: clasificador de intención, redacción, barrera numérica, flujo del chat y su evaluación | `scripts/05b_asistente`; `src/assistant`; `data/eval` |
| **Ambos** | App y API, contratos, documentación y configuración | `api/`, `app/`, `scripts/04_app_api`, `docs/`, `README.md`, `CLAUDE.md`, `PLAN.md`, `requirements/`, `configs/`, `tests/` (los de contrato) |

Reglas del reparto:

- **Al iniciar una sesión, si no sabes con quién hablas, pregunta** si es Luis o Ernesto, y trabaja solo en el frente de esa persona.
- **No edites las carpetas propias de la otra persona.** Si tu tarea lo necesita, propón el cambio y dime a quién hay que pedírselo.
- **Cada persona decide sola dentro de su frente.** No mandes a revisión de la otra persona las decisiones de un frente. Se dividieron la carga para trabajar en paralelo, y lo que choque al unir los trabajos se resuelve en ese momento.
- Los dos frentes se hablan **solo por contratos** (`docs/contratos/`): el formato del paquete de evidencia y el del servicio de chat. Si una decisión de tu frente cambia lo que el contrato describe, actualiza el contrato y avísame en una línea qué cambió.
- Mientras la evidencia real no exista, el agente trabaja contra **ejemplos simulados** (`data/mock/`) servidos por `src/evidence/` con `evidence_mode = "mock"` en `configs/config.json`.

---

## 3. Estado y alcance actual

- **Luis:**
  - Fases 0, 0b, 1 y 2 terminadas.
  - Fase 3 (S4, consistencia y explicabilidad) hecha. Falta su revisión general para cerrarla.
- **Ernesto:** Fase 5B (agente) en curso, contra los ejemplos simulados.
- Las Fases 4 (API y app), 5A (evidencia real y RAG) y 6 (cierre) **no se empiezan hasta que se indique explícitamente.**
- El estado detallado está en `PLAN.md` §1.

---

## 4. Reglas duras

1. **Python 3.10 con pip, archivos en UTF-8.** Abre y guarda siempre con `encoding="utf-8"`. No uses nada que exija 3.11 o superior (por ejemplo `tomllib` o `typing.Self`). Si en la máquina hay una versión más nueva, se puede desarrollar en ella, pero el código debe seguir siendo compatible con 3.10; avísame qué versión hay.
2. **Solo las librerías del plan**, instaladas por capas desde `requirements/`:
   - `base.txt`: pandas, numpy, scikit-learn, plotly (+ herramientas mínimas de desarrollo: pytest).
   - `ml.txt`: lo anterior + xgboost, shap, optuna.
   - `lock.txt`: versiones fijadas de base + ml que instalan desde Python 3.10; es lo que se instala normalmente.
   - `app.txt` (se crea en la Fase 4): streamlit, fastapi, pydantic.
   - `assistant.txt` (se crea en la Fase 5): chromadb, ollama (el paquete de Python es solo el cliente; **Ollama es un programa aparte** que se instala en la máquina).
   - Al crear una capa nueva, sus versiones se agregan a `lock.txt`.
   - **Pandera no se usa.** Las validaciones se escriben con pandas puro.
   - **matplotlib no se usa.** Todas las gráficas son con Plotly, incluidas las de SHAP (se construyen con los valores SHAP).
   - Si hace falta una librería mínima que no está aquí (por ejemplo, un servidor para correr FastAPI), **pregúntame primero**.
3. **Todo local.** Sin APIs ni servicios externos de terceros, y ningún dato sale de la máquina. El modelo de lenguaje es Llama 3.1 8B con Ollama local. (La API interna con FastAPI corre en localhost.)
4. **Docker es una posibilidad, no un supuesto.** Nada puede depender de Docker: todo debe correr con `python -m venv` + `pip install -r requirements/...`.
5. **Todo es código en scripts de Python (`.py`), no notebooks.** El repo no tiene archivos `.ipynb`. Cada paso es un script que se corre **completo** desde la raíz del repo (ver §9).
6. **Semilla 42** en todo lo aleatorio. `set_seed()` fija Python y NumPy; XGBoost y Optuna **no** tienen semilla global, así que se les pasa la semilla como parámetro explícito.
7. **Datos fuera de git:** `data/raw/`, `data/processed/`, `data/chroma/`, `reports/`, `scratch/` y los binarios de `models/` van en `.gitignore`. Nunca subas el CSV de datos. De `models/` sí se suben `metadata.json` y `actual.json`.
8. **Formatos sin dependencias extra:** tablas en **CSV** (no parquet: `pyarrow` no está en el plan), parámetros en **JSON** (`configs/config.json`; no YAML: `pyyaml` no está en el plan) y gráficas en **HTML** con `fig.write_html()` (no PNG: haría falta `kaleido`).
9. **Sin fuga de información:**
   - La partición es **temporal por vintage**, nunca aleatoria por fila.
   - Codificadores, percentiles y calibración se ajustan sin ver el conjunto de prueba.
   - El conjunto de prueba se usa **una sola vez**, al final.
10. **Las columnas de resultado nunca son variables de entrada** (ver §5).
11. **Ninguna cifra inventada.** Todo número que se reporte sale de un cálculo ejecutado en el repo. No generes datos sintéticos por tu cuenta; si falta el CSV de datos, avísame y espera.
12. **Sin fechas ni duraciones de calendario en la documentación.** El avance se mide por tareas terminadas, no por tiempo.

---

## 5. Datos (contrato)

**Archivo:** `data/raw/base_cards_sintetica.csv` (base sintética, ~1.09 millones de filas, ~193 MB). Cada persona lo pone en esa ruta; **no va a git**.

**Qué es una fila:** el resumen de todas las cuentas que comparten `VINTAGE`, `STRATEGY`, `STRATEGY_SEGMENT`, `PRODUCT`, `CHANNEL`, `AGE_TAG` y `SCORE_TAG`, observadas en un mes de desempeño `MOB`. Esas **8 columnas son la llave única** y no deben repetirse.

**Corte estático a septiembre de 2026 (202609).** El MOB máximo de una vintage son los meses entre ella y 202609 (202401 → 32, 202402 → 31, …, 202608 → 1, 202609 → 0).

### Nombres de columna

El diccionario trae para cada variable un alias `dummiN` y un nombre real. El CSV actual trae los **nombres reales**. El cargador (`src/data/load.py`) acepta **ambos**, renombra internamente a los **nombres reales** e imprime cuál encontró.

| Nombre real | Alias | Grupo |
|---|---|---|
| VINTAGE | dummi1 | Dimensión |
| STRATEGY | dummi12 | Dimensión |
| STRATEGY_SEGMENT | dummi13 | Dimensión |
| PRODUCT | dummi6 | Dimensión |
| CHANNEL | dummi3 | Dimensión |
| AGE_TAG | dummi2 | Dimensión |
| SCORE_TAG | dummi10 | Dimensión |
| MOB | *(sin alias en el diccionario)* | Dimensión |
| ACCOUNTS | dummi15 | Volumen y exposición |
| CREDITLINESAMTS | dummi16 | Volumen y exposición |
| ENRACCTS | dummi17 | Volumen y exposición |
| ENRAMTS | dummi23 | Volumen y exposición |
| OVER30ACCTS | dummi18 | Resultado |
| OVER30AMTS | dummi19 | Resultado |
| GCLAMTS | dummi20 | Resultado |
| NCLAMTS | dummi21 | Resultado |
| RECOVERYAMTS | dummi22 | Resultado |
| REVENUE | dummi44 | Resultado |
| EXPENSES_ACQ | dummi46 | Resultado |
| EXPENSES_ECM_VAR | dummi47 | Resultado |
| EXPENSES_ECM_FIX | dummi48 | Resultado |
| EXPENSES_TOTAL | dummi49 | Resultado |

`MOB` no tiene alias en el diccionario: es la columna llamada `MOB`. El mapa nombre real ↔ alias se guarda en `data/contracts/column_map.json`.

Las columnas de **resultado** solo construyen objetivos; **nunca son variables de entrada**.

### Qué significa cada columna (confirmado por Luis)

- **Acumuladas** (cuentan lo ocurrido **hasta ese MOB**, no solo en ese mes): `OVER30ACCTS`, `OVER30AMTS`, `GCLAMTS`, `NCLAMTS`, `RECOVERYAMTS` y `REVENUE`.
- **Constantes en la cohorte:** `ACCOUNTS`, `CREDITLINESAMTS` y `EXPENSES_ECM_FIX`. **Del mes:** `ENRACCTS`, `ENRAMTS`, `EXPENSES_ECM_VAR` y `EXPENSES_TOTAL`. `EXPENSES_ACQ` solo existe en MOB = 0.
- `REVENUE` y `EXPENSES_TOTAL` son **de las cuentas que se fueron a Over30**, como dice el diccionario. La base sintética no lo cumple, y la discrepancia está documentada. `EXPENSES_TOTAL = ACQ + VAR + FIX` sí se cumple. Nada de esto afecta S1 ni S3.
- La evidencia de cada punto está en `docs/datos_fase1.md`.

### Estrategia según vintage

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

### Reglas de validación

El validador devuelve un reporte legible: qué falló, en qué fila y qué hacer. Sale con código de error si hay fallas bloqueantes.

- *Bloquean la carga:* llave repetida; columna faltante; valor nulo en una columna del contrato; categoría fuera de catálogo; estrategia que no corresponde a su vintage; MOB mayor al posible para su vintage; valor fuera del rango del diccionario; `NCLAMTS ≠ GCLAMTS − RECOVERYAMTS`; `EXPENSES_TOTAL ≠ ACQ + VAR + FIX`; `EXPENSES_ACQ ≠ 0` con MOB ≠ 0. (Las igualdades se comparan con una tolerancia numérica pequeña.)
- *Solo avisan:* `OVER30ACCTS > ENRACCTS`; una columna acumulada (`OVER30ACCTS`, `REVENUE`) que baja al subir el MOB dentro de una cohorte; columnas no declaradas en el diccionario (se descartan).

**Detalle del diccionario:** dos valores de catálogo traen un espacio al final (`"Brands-NonPayroll-OffUs&LE "` y `"AAC Off Us Non-Payroll "`), y los datos también. Al cargar se aplica `.str.strip()` a todas las dimensiones, y `src/data/catalog.py` guarda los valores sin espacios.

---

## 6. Definición de las salidas

- **S1:** objetivo = `CREDITLINESAMTS / ACCOUNTS` en **MOB = 0** (la línea que se asigna al booking). XGBoost sobre el **logaritmo** de la línea, con peso = `ACCOUNTS`. Líneas base: media ponderada por producto y score; GLM Gamma. **Publicado:** la media ponderada por producto × score, porque el XGBoost quedó 2.9% peor en MAE de prueba (decisión de Luis). Se explica con su descomposición exacta: promedio global × factor del producto × factor del score dentro del producto.
- **S2:** `S2 = cuentas planeadas × S1`. Función determinista, no modelo.
- **S3:** objetivo = `OVER30ACCTS / ENRACCTS` en **MOB = H = 6**, como etiqueta fraccional con peso = `ENRACCTS`, efecto **decreciente** del score (restricción monótona) y calibración para que la tasa predicha coincida con la observada. Líneas base: tasa ponderada por score y producto; GLM binomial. Como `OVER30ACCTS` es acumulado, S3 es la **tasa acumulada de Over30 a 6 meses**. `OVER30ACCTS > ENRACCTS` no ocurre en ninguna cohorte, así que el denominador es `ENRACCTS`. Se explica con SHAP en escala logit, sobre la salida **sin calibrar**.
- **S4:** comparación de escenarios; se llama *sugerida*, no *óptima*. Solo con la ventana de asignación aleatoria (r.Reca champ contra DARIA 1.6) y solo en el soporte común.
  - **Método 1:** diferencia observada con intervalo bootstrap.
  - **Método 2:** tasa predicha por S3 de las cohortes reales de cada estrategia, dejando fuera una vintage a la vez.
  - Se sugiere solo si el intervalo del método 1 excluye cero y los dos métodos coinciden en el signo; si no, "Sin diferencia concluyente".

---

## 7. Variables de entrada, partición y criterios

**Entran:** STRATEGY, STRATEGY_SEGMENT, PRODUCT y CHANNEL en one-hot; AGE_TAG como ordinal (1 a 3); el score como **posición relativa entre 0 y 1 dentro de su familia** más la familia (Recalibration o DARIA); y el **mes de booking (1–12)** derivado de la vintage, para capturar estacionalidad.

**No entran:** la vintage como número, el MOB, ACCOUNTS y ENRACCTS (son pesos) ni ninguna columna de resultado.

**Partición temporal por vintage:**

| Conjunto | Vintages | Uso |
|---|---|---|
| Entrenamiento | 202401–202508 (20) | Ajustar el modelo |
| Validación | 202509–202511 (3) | Hiperparámetros y calibración |
| Prueba | 202512–202603 (4); para S1, hasta 202609 | Evaluación final, una sola vez |

DARIA aparece solo en validación y prueba, así que esa parte mide extrapolación: **las métricas se reportan por estrategia**. Al aprobar los criterios, cada modelo se reentrena con todas las vintages elegibles.

**Configuración común de XGBoost:** Optuna con 50 pruebas, parada temprana de 50 rondas, semilla 42 y árboles poco profundos (hay pocas vintages). **Un XGBoost se publica solo si supera a su línea base.**

| Salida | Métricas | Criterio en prueba |
|---|---|---|
| S1 | MAE, MAPE, R² y sesgo, ponderados | MAE al menos 10% menor que el de la línea base simple |
| S3 | AUC, KS, Gini, log loss y Brier ponderados; calibración por deciles | No peor que el GLM en log loss y AUC; razón observado/predicho entre 0.95 y 1.05; tasa observada creciente por nivel de riesgo |
| S4 | Signo y ancho del intervalo; concordancia entre métodos | Mismo signo en al menos 70% de los perfiles con diferencia concluyente |

Para AUC y calibración, cada fila aporta tantos eventos como cuentas en Over30 y tantos no eventos como cuentas expuestas restantes. S3 también se reporta por rango de edad.

**Niveles de riesgo:** Bajo hasta P50; Medio P50–P80; Alto P80–P95; Crítico sobre P95. Los percentiles son de la **tasa predicha calibrada en entrenamiento**, ponderada por `ENRACCTS`, y están guardados en el `metadata.json` de S3. La propuesta pedía la tasa histórica observada; es una desviación aprobada, porque el 35% de las cohortes tiene tasa observada 0.

**Confianza** según las cuentas maduras en perfiles similares (misma familia, segmento, producto, canal y edad): Alta con 1,000 o más, Media de 200 a 999 y Baja con menos de 200. En DARIA 2.0 es "No aplica", porque no tiene tasa.

**Fuera de dominio:** un perfil cuya combinación (familia, segmento, producto, canal) no aparece en el entrenamiento de un modelo no recibe la salida de ese modelo, y sale con estado y motivo.

**Versionado:** cada modelo se guarda con un `metadata.json`: fecha, vintages usadas, variables, hiperparámetros, métricas, **versión de datos (hash SHA-256 del CSV)**, umbrales de riesgo y versiones de librerías.

---

## 8. Estructura del repo

```
├── CLAUDE.md  PLAN.md  README.md
├── configs/config.json            # semilla, H, particiones, rutas, evidence_mode, Ollama, Chroma
├── requirements/                  # base.txt, ml.txt, lock.txt (app.txt y assistant.txt en Fases 4 y 5)
├── data/
│   ├── raw/                       # base_cards_sintetica.csv (NO git)
│   ├── contracts/                 # diccionario y column_map.json (sí git)
│   ├── processed/                 # tablas derivadas en CSV (NO git)
│   ├── mock/                      # paquetes de evidencia de ejemplo (sí git)
│   ├── knowledge/                 # documentos base del RAG, en markdown (sí git)
│   ├── eval/                      # preguntas de evaluación del asistente (sí git)
│   └── chroma/                    # índice vectorial (NO git)
├── scripts/                       # pasos ejecutables, numerados
│   ├── run_fase.py  _plantilla.py
│   ├── 01_datos/  02_modelos_prediccion/  03_estrategia_y_explicabilidad/
│   ├── 04_app_api/  05a_evidencia_rag/  05b_asistente/  06_cierre/
├── src/                           # funciones reutilizables
│   ├── utils/  data/  features/  models/  strategy/  consistency/
│   ├── explain/  evidence/  rag/  assistant/
├── models/                        # <salida>/<versión>/{modelo.*, metadata.json} y actual.json
├── reports/                       # tablas y gráficas generadas por los scripts (NO git)
├── api/  app/                     # FastAPI y Streamlit
├── docs/                          # propuesta, contratos/, fichas/, decisiones por fase, manuales
├── scratch/                       # pruebas personales: scratch/luis/, scratch/ernesto/ (NO git)
└── tests/
```

**Regla clave:** los **scripts** son los pasos del pipeline (explican, ejecutan, guardan resultados); la **lógica reutilizable** vive en `src/` para que la API, la app y el agente la importen sin copiarla.

---

## 9. Convenciones de scripts y código

### Scripts (`scripts/<fase>/NN_nombre.py`)

- Se corren **completos** desde la raíz del repo: `python scripts/01_datos/01_carga_y_diccionario.py`. `python scripts/run_fase.py 1` corre en orden los scripts de una fase.
- **Estructura estándar** (copia `scripts/_plantilla.py`):
  1. Docstring inicial: título, **objetivo**, **entradas**, **salidas**, **cómo correrlo** y **listo cuando**.
  2. Encabezado estándar para poder importar `src` desde la raíz:
     `import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`
  3. Secciones numeradas con comentarios de bloque (`# ===== 1. Cargar datos =====`).
  4. Una función `main()`; el cálculo y el guardado primero, las **gráficas en funciones aparte** al final.
  5. `argparse` con `--sin-graficas` (y los parámetros que hagan falta); `--help` debe explicar el script.
- Cada script debe correr **de principio a fin sin intervención**, dar el **mismo resultado** si se repite (semilla fija) y **salir con código distinto de 0** si falla algo bloqueante.
- Imprime en terminal un **resumen claro** de lo que hizo (cuántas filas, qué reglas pasaron, qué archivos escribió).
- **Salidas:** datasets en `data/processed/`; tablas (CSV) y gráficas (HTML) en `reports/<fase>/`; modelos en `models/`. Las rutas se leen de `configs/config.json`, nunca se escriben a mano.

### Documentación del código

- **Qué hace → por qué.** Al comentar o explicar código, primero algo objetivo (qué guarda cada variable, con qué función y cómo) y **después** el fundamento.
- **Fórmula → código.** Donde haya una fórmula, un comentario que la traduce símbolo por símbolo: qué variable del código representa cada término. Asume una base matemática sólida; lo que hace falta ver es cómo se mapea a las líneas, no la matemática desde cero.
- Funciones pequeñas, con nombre claro y sin estado global. Nombres de funciones y variables en inglés (`snake_case`); texto, comentarios y docstrings en **español**. Docstring corto (qué hace, parámetros, qué devuelve) y type hints básicos.
- Toda función que se use en más de un script lleva una prueba en `tests/` (`pytest`).

---

## 10. Cómo quiero que trabajes

- Avanza **un script a la vez** y dime qué hiciste y qué sigue antes de pasar al siguiente. No saltes de fase sin mi visto bueno.
- Al terminar una tarea, marca su casilla en `PLAN.md` y **actualiza `README.md`** si agregaste un script o una carpeta.
- Si modificas un archivo que ya existe, **cambia solo lo necesario** y dime qué cambiaste y dónde; no lo reescribas completo.
- Si algo es ambiguo en los datos o en la propuesta, no lo resuelvas en silencio: documéntalo, propón una decisión y pregúntame.
- Si una librería no instala en Python 3.10, dímelo antes de buscar alternativas.

---

## 11. Git

- Un commit por tarea terminada (por ejemplo, un script), con mensaje corto en español y en imperativo: `Agrega validación del contrato de datos`.
- Antes de cada commit, revisa `git status` y confirma que no entran `data/` (salvo `contracts`, `mock`, `knowledge`, `eval`), `reports/`, `scratch/`, binarios de `models/` ni `.venv/`. No uses `git add .` a ciegas.
- Una rama por tarea o fase, con el nombre de la persona: `luis/fase-1-datos`, `ernesto/asistente-clasificador`. No trabajes directo en `main`. Antes de empezar una sesión, haz `git pull`. Los cambios llegan a `main` por *pull request* revisado por la otra persona.
- Pídeme confirmación antes de `git push` y de borrar ramas. Por ahora los `push` los hago yo.

### Comandos prohibidos (aun con mi confirmación)

**Nunca ejecutes, ni me pidas permiso para ejecutar:**
- `git reset --hard`
- cualquier `git push --force`, `-f` o `--force-with-lease`
- sus equivalentes: `git clean -fd`, `git checkout -- .` o `git restore .` sobre cambios sin guardar, `git rebase` de ramas ya subidas y `git filter-branch`

Si crees que uno de ellos es la única salida, **no lo ejecutes y no me pidas autorización**. Explícame: cuál es el problema, por qué crees que no hay otra salida, qué se perdería o se reescribiría, y qué alternativas más seguras descartaste y por qué. **Solo puedes sugerírmelo**, y lo ejecutaría yo manualmente si me convence.

Estos comandos además están **bloqueados técnicamente** en `.claude/settings.json` (ver `PLAN.md`, Fase 0b).

---

## 12. Decisiones confirmadas y preguntas abiertas

Cada decisión tiene su evidencia en el documento que se indica.

**Proyecto**
- El pipeline es de scripts `.py`; no hay notebooks. Python 3.10 con pip y UTF-8; solo las librerías del plan; todo local; Docker es opcional.
- Luis: datos, modelos, consistencia, explicabilidad, evidencia y RAG. Ernesto: el agente. Cada uno decide dentro de su frente (§2).

**Datos** (`docs/datos_fase1.md`)
- **El diccionario de datos es la fuente de verdad (exacto al 100%).** Si la base sintética no lo refleja, manda el diccionario: el código, los documentos del RAG y las respuestas del agente usan sus definiciones. Nada debe depender de un comportamiento de la base sintética que el diccionario no garantice, porque los datos reales seguirán el diccionario.
- "En el mes de información de performance MOB" significa *observado al corte de ese MOB* (acumulado hasta ese MOB). Por eso S3 se lee directamente en MOB = 6.
- El CSV trae nombres reales y `MOB` se llama `MOB`. Las columnas acumuladas, constantes y del mes están en §5.
- Un valor nulo en cualquier columna del contrato es falla bloqueante: no se rellena ni se descarta en silencio.
- Ningún dato se descarta de la base: los conjuntos de modelado son subconjuntos derivados y la base completa queda disponible (incluido el RAG).
- El denominador de S3 es `ENRACCTS`: `OVER30ACCTS > ENRACCTS` ocurre en 0 de 52,835 cohortes.

**Modelos** (`docs/fichas/`)
- S1 publicado = línea base simple (producto × score), porque el XGBoost no cumplió el criterio. El XGBoost queda guardado como registro.
- S3 publicado = XGBoost con calibración Platt. Los umbrales de riesgo son percentiles de la tasa predicha en entrenamiento (desviación aprobada, §7).

**Estrategia y consistencia** (`docs/consistencia_fase3.md`)
- S4 compara r.Reca champ contra DARIA 1.6 solo en el **soporte común**; fuera de él, "Sin evidencia". DARIA 2.0: "No evaluable en riesgo".
- Método 2 de S4: S3 reentrenado dejando fuera una vintage a la vez, para no inflar la concordancia.
- "Fuera de dominio", "perfiles similares" y la confianza "No aplica" de DARIA 2.0 están definidos en §7.

**Explicabilidad**
- S3 se explica con el TreeSHAP nativo de XGBoost (`pred_contribs`), porque shap 0.49.1 no lee XGBoost 3.x; es el mismo algoritmo. S1 se explica con su descomposición exacta por factores.

**Por decidir en la Fase 5** (se preguntan al llegar): el modelo de embeddings local de Chroma.

---

## 13. Glosario mínimo

- **Booking:** momento en que se abre la cuenta. **Vintage:** mes y año del booking (`AAAAMM`).
- **MOB (month on books):** mes de desempeño relativo al booking; MOB 0 es el mes del booking.
- **Cohorte / perfil:** conjunto de cuentas con las mismas 7 dimensiones.
- **Over30:** cuentas que se fueron a mora de más de 30 días (definición habitual; confirmar la exacta con los datos). `OVER30ACCTS` es el acumulado hasta ese MOB.
- **GCL / NCL:** pérdida bruta / pérdida neta (`NCL = GCL − recuperación`).
- **Cuentas expuestas (`ENRACCTS`):** cuentas con saldo; denominador de la tasa de impago.
- **Reca champ / Recalibration / DARIA:** familias de estrategia de adquisición (ver §5).
- **Paquete de evidencia:** resultado de una función determinista (cifras, tablas, versiones) sobre el que el agente redacta; ver `docs/contratos/evidencia.md`.
