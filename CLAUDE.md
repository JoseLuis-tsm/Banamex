# CLAUDE.md — Sistema de Inteligencia Crediticia (Tarjetas de Crédito)

> Contexto permanente para Claude Code. Léelo completo al iniciar cada sesión.
> - Plan de trabajo y avance: `PLAN.md`
> - Guía de uso del repo (estructura, dónde poner datos, cómo correr): `README.md`
> - Propuesta original: `docs/Propuesta_Tecnica.pdf`
> - Diccionario de variables: `data/contracts/diccionario_variables.csv`
> - Contratos entre frentes: `docs/contratos/`
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

**Usuario final:** personal del banco sin especialización en riesgo de crédito ni ciencia de datos. Todo lo que vea debe ser claro y en español. Los analistas de riesgo y la gerencia usan una vista avanzada y una ejecutiva de la misma herramienta.

---

## 2. Equipo, reparto y con quién hablas

Dos personas trabajan **en paralelo** en el mismo repo:

| Persona | Frente | Carpetas propias |
|---|---|---|
| **Luis** | Datos, modelos (S1, S2, S3, S4), consistencia, SHAP, evidencia determinista y RAG | `scripts/01_datos`, `02_modelos_prediccion`, `03_estrategia_y_explicabilidad`, `05a_evidencia_rag`; `src/data`, `features`, `models`, `strategy`, `consistency`, `explain`, `evidence`, `rag`; `data/knowledge`; `models/` |
| **Ernesto** | El agente: clasificador de intención, redacción, barrera numérica, flujo del chat y su evaluación | `scripts/05b_asistente`; `src/assistant`; `data/eval` |
| **Ambos** | App y API, contratos, documentación y configuración | `api/`, `app/`, `scripts/04_app_api`, `docs/`, `README.md`, `CLAUDE.md`, `PLAN.md`, `requirements/`, `configs/`, `tests/` (los de contrato) |

Reglas del reparto:

- **Al iniciar una sesión, si no sabes con quién hablas, pregunta** si es Luis o Ernesto, y trabaja solo en el frente de esa persona.
- **No edites las carpetas propias de la otra persona.** Si tu tarea lo necesita, propón el cambio y dime a quién hay que pedírselo.
- Los dos frentes se hablan **solo por contratos** (`docs/contratos/`): el formato del paquete de evidencia y el del servicio de chat. Cambiar un contrato requiere que ambos estén de acuerdo; avísame antes de tocarlo.
- Mientras la evidencia real no exista, el agente trabaja contra **ejemplos simulados** (`data/mock/`) servidos por `src/evidence/` con `evidence_mode = "mock"` en `configs/config.json`.

---

## 3. Estado y alcance actual

- **Luis:** Fase 0b (reestructura) → Fase 1 (datos) → Fase 2 (modelos S1, S2 y S3).
- **Ernesto:** Fase 5B (agente) contra los ejemplos simulados, en cuanto la Fase 0b esté terminada.
- Todo lo demás (S4, SHAP, evidencia real, RAG, API, app, cierre) **no se empieza hasta que se indique explícitamente.**

---

## 4. Reglas duras

1. **Python 3.10 con pip, archivos en UTF-8.** Abre y guarda siempre con `encoding="utf-8"`. No uses nada que exija 3.11 o superior (por ejemplo `tomllib` o `typing.Self`). Si en la máquina hay una versión más nueva, se puede desarrollar en ella, pero el código debe seguir siendo compatible con 3.10; avísame qué versión hay.
2. **Solo las librerías del plan**, instaladas por capas desde `requirements/`:
   - `base.txt`: pandas, numpy, scikit-learn, plotly (+ herramientas mínimas de desarrollo: pytest).
   - `ml.txt`: lo anterior + xgboost, shap, optuna.
   - `app.txt`: streamlit, fastapi, pydantic.
   - `assistant.txt`: chromadb, ollama (el paquete de Python es solo el cliente; **Ollama es un programa aparte** que se instala en la máquina).
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

El diccionario trae para cada variable un alias `dummiN` y un nombre real. Luis indicó que **los datos vienen con los alias `dummiN`**; el reporte de la Fase 0 decía que el CSV traía los nombres reales. **Verifícalo con el encabezado del CSV.** El cargador debe aceptar **ambos** y renombrar internamente a los **nombres reales**, imprimiendo cuál encontró.

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

`MOB` no tiene alias en el diccionario: busca una columna llamada `MOB` y, si no queda claro cuál es, pregúntame. El mapa nombre real ↔ alias se guarda en `data/contracts/column_map.json`.

Las columnas de **resultado** solo construyen objetivos; **nunca son variables de entrada**.

### Qué significa cada columna (confirmado por Luis)

- **`OVER30ACCTS` es acumulado:** cuenta las cuentas que llegaron a Over30 **hasta ese MOB**, no solo las de ese mes.
- **`REVENUE` también es acumulado** (según Luis; confírmalo con los datos).
- Para las demás columnas de resultado (`OVER30AMTS`, `GCLAMTS`, `NCLAMTS`, `RECOVERYAMTS`, `EXPENSES_*`) **comprueba con los datos** si son acumuladas o del mes (dentro de una cohorte, una serie acumulada nunca baja al subir el MOB) y documenta la evidencia.
- Las descripciones de `REVENUE` y `EXPENSES_TOTAL` dicen "de las cuentas que se fueron a Over30", pero las de sus componentes de gasto no. Comprueba que `EXPENSES_TOTAL = EXPENSES_ACQ + EXPENSES_ECM_VAR + EXPENSES_ECM_FIX` se cumpla y documenta a qué población se refiere. Esto no afecta S1 ni S3.

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

**Detalle del diccionario:** algunos valores de catálogo traen un espacio al final (por ejemplo `"Brands-NonPayroll-OffUs&LE "`, `"AAC Off Us Non-Payroll "`). Compara contra los datos y normaliza con `.str.strip()` si procede, documentando la decisión.

---

## 6. Definición de las salidas

- **S1:** objetivo = `CREDITLINESAMTS / ACCOUNTS` en **MOB = 0** (la línea que se asigna al booking). XGBoost sobre el **logaritmo** de la línea, con peso = `ACCOUNTS`. Líneas base: media ponderada por producto y score; GLM Gamma.
- **S2:** `S2 = cuentas planeadas × S1`. Función determinista, no modelo.
- **S3:** objetivo = `OVER30ACCTS / ENRACCTS` en **MOB = H = 6**, como etiqueta fraccional con peso = `ENRACCTS`, efecto **decreciente** del score (restricción monótona) y calibración para que la tasa predicha coincida con la observada. Líneas base: tasa ponderada por score y producto; GLM binomial. Como `OVER30ACCTS` es acumulado, S3 es la **tasa acumulada de Over30 a 6 meses**. Verifica en la Fase 1 en qué proporción de cohortes `OVER30ACCTS > ENRACCTS` en MOB = 6 (el numerador es acumulado y el denominador es el saldo vigente); si es frecuente, **propón y pregúntame** usar `ACCOUNTS` como denominador.
- **S4:** (Fase 3) comparación de escenarios; se llama *sugerida*, no *óptima*. Solo con la ventana de asignación aleatoria, y solo se sugiere si el intervalo excluye cero y los dos métodos coinciden en el signo; si no, "Sin diferencia concluyente".

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
| S3 | AUC, KS, Gini, log loss y Brier ponderados; calibración por deciles | No peor que el GLM; razón observado/predicho entre 0.95 y 1.05; tasa observada creciente por nivel de riesgo |
| S4 | Signo y ancho del intervalo; concordancia entre métodos | Mismo signo en al menos 70% de los perfiles con diferencia concluyente |

Para AUC y calibración, cada fila aporta tantos eventos como cuentas en Over30 y tantos no eventos como cuentas expuestas restantes. S3 también se reporta por rango de edad.

**Niveles de riesgo** (percentiles de la tasa histórica ponderada): Bajo hasta P50; Medio P50–P80; Alto P80–P95; Crítico sobre P95. **Confianza** según cuentas maduras en perfiles similares: Alta con 1,000 o más, Media de 200 a 999, Baja con menos de 200.

**Versionado:** cada modelo se guarda con un `metadata.json`: fecha, vintages usadas, variables, hiperparámetros, métricas, **versión de datos (hash SHA-256 del CSV)**, umbrales de riesgo y versiones de librerías.

---

## 8. Estructura del repo

```
├── CLAUDE.md  PLAN.md  README.md
├── configs/config.json            # semilla, H, particiones, rutas, evidence_mode, Ollama, Chroma
├── requirements/                  # base.txt, ml.txt, app.txt, assistant.txt, lock.txt
├── data/
│   ├── raw/                       # base_cards_sintetica.csv (NO git)
│   ├── contracts/                 # diccionario, column_map.json, reglas de validación (sí git)
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
├── docs/                          # propuesta, contratos/, fichas de modelo, manuales
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

**Confirmadas**
- Los datos están en `data/raw/base_cards_sintetica.csv`.
- `OVER30ACCTS` y `REVENUE` son acumulados. En el diccionario, "en el mes de información de performance MOB" significa *observado al corte de ese MOB* (acumulado hasta ese MOB), no *solo lo ocurrido en ese mes*. Por eso S3 se lee directamente en MOB = 6.
- El pipeline es de scripts `.py`; no hay notebooks.
- Python 3.10 con pip, UTF-8; solo las librerías del plan; todo local; Docker es opcional.
- Luis: datos, modelos, SHAP, evidencia y RAG. Ernesto: el agente.
- Un valor nulo en cualquier columna del contrato es falla bloqueante (no se rellena ni se descarta en silencio).
- **El diccionario de datos es la fuente de verdad (exacto al 100%).** Si la base sintética no lo refleja, manda el diccionario: el código, los documentos del RAG y las respuestas del agente usan sus definiciones, y nada debe depender de un comportamiento de la base sintética que el diccionario no garantice (los datos reales seguirán el diccionario).
- Por eso `REVENUE` y `EXPENSES_TOTAL` son **de las cuentas que se fueron a Over30**, como dice el diccionario; la discrepancia de la base sintética queda documentada en `docs/datos_fase1.md`.
- S4 compara estrategias solo en el **soporte común** (perfiles con cuentas en ambas); fuera de él, "sin evidencia".
- Ningún dato se descarta de la base: los conjuntos de modelado son subconjuntos derivados y la base completa queda disponible (incluido el RAG).

**Resueltas con los datos (Fase 1; detalle en `docs/datos_fase1.md`)**
1. El CSV trae los nombres reales (el cargador acepta también los alias `dummiN`).
2. `MOB` es la columna llamada `MOB`.
3. Acumuladas: `OVER30ACCTS`, `OVER30AMTS`, `GCLAMTS`, `NCLAMTS`, `RECOVERYAMTS` y `REVENUE`.
4. `OVER30ACCTS > ENRACCTS` en MOB = 6 en 0 de 52,835 cohortes: se mantiene `ENRACCTS` como denominador de S3.

**Por decidir en la Fase 5** (se preguntan al llegar): embeddings de Chroma local y definición exacta de "perfil fuera de dominio" y de "perfiles similares" para la confianza.

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
