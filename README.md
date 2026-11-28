# Sistema de Inteligencia Crediticia (Tarjetas de Crédito)

Plataforma **local** que, para un perfil de adquisición de tarjetas de crédito, estima cuatro salidas:

| Salida | Pregunta que responde | Cómo se obtiene |
|---|---|---|
| **S1.** Línea promedio esperada | ¿Qué límite promedio recibe una cuenta de este perfil? | Tabla de medias ponderadas por producto × score (publicada) |
| **S2.** Monto total esperado | ¿Cuánto crédito implica colocar N cuentas? | Cálculo: cuentas planeadas × S1 |
| **S3.** Tasa de impago esperada | ¿Qué fracción de cuentas estará en Over30 a 6 meses? | XGBoost calibrado (publicado) |
| **S4.** Estrategia sugerida | ¿Qué estrategia rinde mejor para este perfil? | Comparación de r.Reca champ contra DARIA 1.6 con intervalo |

Cada resultado se explica (SHAP en S3; factores exactos en S1) y se podrá consultar en lenguaje natural con un asistente (RAG + Ollama) que **solo redacta sobre cifras ya calculadas**.

**No** aprueba ni rechaza créditos, **no** predice por cliente individual y **no** afirma causalidad. Los datos del proyecto son una **base sintética**.

| Documento | Para qué |
|---|---|
| **README.md** (este) | Instalar, correr y encontrar cada cosa |
| [PLAN.md](PLAN.md) | Qué se hizo, qué falta y con qué criterio se da por terminado |
| [CLAUDE.md](CLAUDE.md) | Reglas del proyecto y decisiones confirmadas (contexto para Claude Code) |

---

## 1. Estado

| Fase | Qué produce | Estado |
|---|---|---|
| 0 y 0b. Preparación | Repo, entorno, scripts, contratos y ejemplos simulados | ✅ |
| 1. Datos | Base validada y datasets de S1 y S3 | ✅ |
| 2. Modelos S1, S2 y S3 | S1 (línea base simple) y S3 (XGBoost) publicados; S2 como cálculo | ✅ |
| 3. Estrategia, consistencia y explicabilidad | S4, reglas de consistencia, SHAP de S3 y factores de S1 | ✅ |
| 4. App y API | FastAPI + Streamlit | ⬜ |
| 5A. Evidencia y RAG | Funciones de evidencia C1–C7 y base documental | ⬜ |
| 5B. Agente (Ernesto) | Clasificador, redacción, barrera numérica, chat | En curso, con ejemplos simulados |
| 6. Cierre | Pruebas de extremo a extremo y criterios de aceptación | ⬜ |

El detalle de cada fase está en [PLAN.md](PLAN.md).

---

## 2. Instalación

**Requisitos:** Python 3.10 o superior con pip, y Git. Para el asistente, además, **Ollama** con el modelo `llama3.1:8b` (es un programa aparte, no un paquete de Python). Nada depende de Docker ni de servicios externos.

### Windows (PowerShell)

```powershell
git clone https://github.com/JoseLuis-tsm/Banamex.git
cd Banamex
python -m venv .venv
# Solo la primera vez, si PowerShell no deja activar el entorno:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.venv\Scripts\Activate.ps1
python -m pip install -r requirements/lock.txt
python -m pytest
```

### macOS / Linux

```bash
git clone https://github.com/JoseLuis-tsm/Banamex.git
cd Banamex
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/lock.txt
python -m pytest
```

`python -m pytest` debe terminar con todas las pruebas en verde.

**Librerías** (`requirements/`), por capas:

| Archivo | Contiene | Se usa en |
|---|---|---|
| `base.txt` | pandas, numpy, scikit-learn, plotly, pytest | Todo |
| `ml.txt` | base + xgboost, shap, optuna | Fases 2 y 3 |
| `lock.txt` | Versiones fijadas de base + ml (instalan desde Python 3.10) | **La instalación normal** |
| `app.txt`, `assistant.txt` | streamlit, fastapi, pydantic · chromadb, ollama | Se crean al empezar las Fases 4 y 5 |

**Ollama (solo para el asistente):** instala Ollama desde ollama.com, corre `ollama pull llama3.1:8b` y comprueba con `ollama list`. El servidor local queda en `http://localhost:11434` (sección `ollama` de `configs/config.json`).

---

## 3. Primer uso

### 3.1 Pon los datos

Copia el CSV a **`data/raw/base_cards_sintetica.csv`**, con ese nombre exacto. No va a git: el `.gitignore` lo excluye. Si no lo tienes, pídeselo a Luis.

- **Luis** necesita el CSV para correr las Fases 1 a 3.
- **Ernesto** no lo necesita para el agente: trabaja con los ejemplos simulados de `data/mock/`, que sirve `get_evidence()` mientras `evidence_mode = "mock"` en `configs/config.json`. Cuando la evidencia real esté lista (Fase 5A), `evidence_mode` pasa a `"real"` y entonces sí hace falta el CSV.

### 3.2 Corre el pipeline

Siempre desde la **raíz del repo** y con el entorno activado.

| Qué | Comando |
|---|---|
| Una fase completa, en orden | `python scripts/run_fase.py 1` (fases: 1, 2, 3, 4, 5a, 5b, 6) |
| Continuar una fase desde un script | `python scripts/run_fase.py 3 --desde 05` |
| Un solo script | `python scripts/01_datos/01_carga_y_diccionario.py` |
| Sin gráficas (más rápido) | agrega `--sin-graficas` |
| Ayuda de un script | `python scripts/<fase>/<script>.py --help` |
| Todas las pruebas | `python -m pytest` |

Para reconstruir todo desde cero, corre en orden `run_fase.py 1`, `2` y `3`. Cada fase lee lo que escribió la anterior.

`run_fase.py` se detiene en el primer error, imprime un resumen (script, resultado y segundos) y dice cómo continuar. Cada script imprime al final qué archivos escribió.

---

## 4. Cómo fluye el proyecto

```
data/raw/base_cards_sintetica.csv
        │
        ▼
 Fase 1 · scripts/01_datos/ ───────────────► data/processed/base_estandar.csv
        │                                    data/processed/s1_dataset.csv, s3_dataset.csv
        ▼
 Fase 2 · scripts/02_modelos_prediccion/ ──► data/processed/s1_modelado.csv, s3_modelado.csv
        │                                    models/s1/, models/s3/   (modelos publicados)
        │                                    docs/fichas/             (fichas de modelo)
        ▼
 Fase 3 · scripts/03_estrategia_y_explicabilidad/
        ├─ S4 (01–04) ─────────────────────► s4_sugerencia.csv, s4_concordancia.csv
        ├─ Consistencia (05) ──────────────► consistencia_*.csv
        └─ Explicabilidad (06–10) ─────────► data/processed/shap_s1.csv, shap_s3.csv y gráficas
                                             (todo lo demás en reports/03_estrategia_y_explicabilidad/)
        ▼
 Fases 4 y 5 · API, app, evidencia C1–C7 y asistente (por hacer)
```

**Qué hace cada carpeta:**

- Los **scripts** (`scripts/`) son los pasos del pipeline: cargan, calculan, guardan y explican.
- La **lógica reutilizable** vive en `src/`, para que la importen los scripts, la API, la app y el agente sin copiarla.
- Los **parámetros** viven en `configs/config.json`: semilla, horizonte, particiones y rutas. Ningún script escribe rutas a mano.

| Paquete de `src/` | Qué contiene | Lo usan |
|---|---|---|
| `utils/` | `load_config`, `get_path`, `set_seed` y el estilo común de gráficas | Todos |
| `data/` | Carga, catálogo del diccionario, validación del contrato y madurez de cohortes | Fase 1 y siguientes |
| `features/` | Variables de entrada y partición temporal | Fases 2 y 3 |
| `models/` | Líneas base, métricas, búsqueda con Optuna, predicción, S2 y registro de versiones | Fases 2 y 3 |
| `strategy/` | Balance entre estrategias y S4 (bootstrap, método 2, regla de decisión) | Fases 1 y 3 |
| `consistency/` | Reglas que corren después de los modelos: dominio, confianza, niveles y alertas | Fase 3 y la API |
| `explain/` | SHAP de S3, factores de S1, frases de lectura y gráficas | Fase 3 y la evidencia |
| `evidence/` | `get_evidence()`; hoy sirve los ejemplos simulados | Agente |
| `rag/`, `assistant/` | Vacíos todavía (Fases 5A y 5B) | — |

---

## 5. Dónde está cada cosa

### 5.1 Lo que más se busca

| Busco… | Está en |
|---|---|
| El modelo publicado de S1 o S3 | `models/<s1 o s3>/actual.json` dice qué versión está vigente. El modelo es `models/<salida>/<versión>/modelo.*` con su `metadata.json`. |
| El respaldo de S1 | `models/s1/s1-4eeb609786/`: el XGBoost de S1, no publicado. Se activa apuntando `models/s1/actual.json` a esa versión. |
| Métricas y explicación de un modelo | [docs/fichas/ficha_s1.md](docs/fichas/ficha_s1.md), [ficha_s3.md](docs/fichas/ficha_s3.md) |
| Hallazgos y decisiones de datos | [docs/datos_fase1.md](docs/datos_fase1.md) |
| Decisiones de S4 y de consistencia | [docs/consistencia_fase3.md](docs/consistencia_fase3.md) |
| Formato de la evidencia y del chat | [docs/contratos/evidencia.md](docs/contratos/evidencia.md), [chat.md](docs/contratos/chat.md) |
| Diccionario de variables | `data/contracts/diccionario_variables.csv` |
| Tablas y gráficas generadas | `reports/<fase>/` (CSV) y `reports/<fase>/graficas/` (HTML, se abren en el navegador) |
| Datasets derivados | `data/processed/` |
| La propuesta original | `docs/Propuesta_Tecnica.pdf` |

Los modelos binarios, `reports/` y `data/processed/` **no van a git**: se regeneran con los scripts. Sí se versionan `metadata.json` y `actual.json`.

### 5.2 Estructura del repo

```
├── README.md  PLAN.md  CLAUDE.md  pytest.ini
├── .claude/settings.json          # bloqueo de comandos de git destructivos para Claude Code
├── configs/config.json            # semilla, horizonte, particiones, rutas, evidence_mode, Ollama
├── requirements/                  # base.txt, ml.txt, lock.txt
├── data/
│   ├── raw/                       # CSV de datos                              (NO git)
│   ├── contracts/                 # diccionario y column_map.json
│   ├── processed/                 # datasets derivados en CSV                 (NO git)
│   ├── mock/                      # ejemplos simulados de evidencia
│   ├── knowledge/                 # documentos del RAG (Fase 5A)
│   ├── eval/                      # preguntas de evaluación del asistente (Fase 5B)
│   └── chroma/                    # índice vectorial, se crea al indexar      (NO git)
├── scripts/
│   ├── run_fase.py  _plantilla.py
│   ├── 01_datos/  02_modelos_prediccion/  03_estrategia_y_explicabilidad/
│   └── 04_app_api/  05a_evidencia_rag/  05b_asistente/  06_cierre/   (por hacer)
├── src/                           # lógica reutilizable (ver §4)
├── models/                        # s1/, s3/: <versión>/{modelo.*, metadata.json} y actual.json
├── reports/                       # tablas y gráficas generadas               (NO git)
├── api/  app/                     # FastAPI y Streamlit (Fase 4)
├── docs/                          # propuesta, contratos/, fichas/, decisiones por fase
├── scratch/                       # pruebas personales: scratch/luis/, scratch/ernesto/ (NO git)
└── tests/                         # pruebas con pytest
```

### 5.3 Quién mantiene qué

| Persona | Frente | Carpetas |
|---|---|---|
| **Luis** | Datos, modelos (S1–S4), consistencia, explicabilidad, evidencia y RAG | `scripts/01`–`03`, `05a`; `src/data`, `features`, `models`, `strategy`, `consistency`, `explain`, `evidence`, `rag`; `data/contracts`, `mock`, `knowledge`; `models/` |
| **Ernesto** | El agente: intención, redacción, barrera numérica, chat y evaluación | `scripts/05b_asistente`; `src/assistant`; `data/eval` |
| **Ambos** | App, API, documentación y configuración | `api/`, `app/`, `scripts/04_app_api`, `06_cierre`; `docs/`, `README.md`, `PLAN.md`, `CLAUDE.md`, `requirements/`, `configs/`, `src/utils`, `tests/` de contrato |

Los dos frentes se comunican por los **contratos** de `docs/contratos/`. Cada quien decide dentro de su frente; lo que choque al unir los trabajos se resuelve en ese momento.

---

## 6. Scripts

Todos se corren desde la raíz. Las salidas sin ruta van en `reports/<fase>/`, y las gráficas en `reports/<fase>/graficas/`.

### Fase 1 · Datos (`scripts/01_datos/`)

| Script | Qué hace | Escribe |
|---|---|---|
| `01_carga_y_diccionario.py` | Carga el CSV (nombres reales o alias `dummiN`), quita espacios a las categorías y verifica el catálogo contra el diccionario | `data/contracts/column_map.json`, `data/processed/base_estandar.csv`, `carga_*.csv` |
| `02_validacion_contrato.py` | Valida el contrato de datos (sale con código 1 si hay fallas bloqueantes) y comprueba qué columnas son acumuladas | `validacion.csv`, `columnas_acumuladas.csv`, `poblacion_columnas.csv` |
| `03_exploracion_cohortes.py` | Cuentas por vintage, estrategia y segmento; producto, canal, score, edad y filas por MOB | `exploracion_*.csv` y gráficas |
| `04_madurez_y_curvas.py` | Madurez a 6 meses, curvas de Over30 y datasets de S1 y S3 | `data/processed/s1_dataset.csv`, `s3_dataset.csv`, `madurez_vintages.csv`, `curvas_*.csv` |
| `05_balance_entre_estrategias.py` | Balance r.Reca champ vs. DARIA 1.6 en la ventana aleatoria y estrategias evaluables | `balance_*.csv`, `estrategias_evaluables.csv` y gráficas |

### Fase 2 · Modelos (`scripts/02_modelos_prediccion/`)

| Script | Qué hace | Escribe |
|---|---|---|
| `00_variables_y_particion.py` | Variables de entrada, partición temporal por vintage y prueba de fuga | `data/processed/s1_modelado.csv`, `s3_modelado.csv`, `particion.csv` |
| `01_s1_linea_base.py` | Líneas base de S1: media por producto × score y GLM Gamma | `s1_linea_base_*.csv` |
| `02_s1_xgboost.py` | XGBoost de S1; se publica solo si su MAE de prueba es ≥ 10% menor que el de la base. No lo fue, así que el script guarda el XGBoost como registro y publica la base simple como S1 | `models/s1/`, `s1_*.csv` y gráficas |
| `03_s2_monto_total.py` | S2 = cuentas planeadas × S1, con comprobación exacta | `s2_monto_total.csv` |
| `04_s3_linea_base.py` | Líneas base de S3: tasa por score × producto y GLM binomial | `s3_linea_base_*.csv` |
| `05_s3_xgboost.py` | XGBoost de S3 (monótono en el score, Optuna, calibración Platt); publica si cumple los criterios | `models/s3/`, `s3_*.csv` y gráficas |
| `06_resumen_modelos.py` | Tabla resumen y fichas de S1 y S3, generadas desde `metadata.json` | `resumen_modelos.csv`, `docs/fichas/` |

### Fase 3 · Estrategia, consistencia y explicabilidad (`scripts/03_estrategia_y_explicabilidad/`)

| Script | Qué hace | Escribe |
|---|---|---|
| `01_ventana_aleatoria.py` | Ventana aleatoria de S4, perfiles producto × canal × edad y soporte común | `data/processed/s4_ventana.csv`, `s4_perfiles.csv`, `s4_balance.csv` |
| `02_s4_observado_bootstrap.py` | S4 método 1: diferencia observada DARIA 1.6 − r.Reca champ con intervalo bootstrap | `s4_metodo1_bootstrap.csv` y gráfica |
| `03_s4_prediccion_escenarios.py` | S4 método 2: tasa predicha por S3, dejando fuera una vintage a la vez | `data/processed/s4_ventana_predicha.csv`, `s4_metodo2_prediccion.csv` y gráfica |
| `04_s4_sugerencia_y_validacion.py` | Regla de decisión y concordancia de signo entre métodos (≥ 70%) | `s4_sugerencia.csv`, `s4_concordancia.csv` y gráfica |
| `05_reglas_consistencia.py` | Reglas de consistencia verificadas al 100% sobre las filas de prueba | `consistencia_prueba.csv`, `consistencia_reglas.csv`, `consistencia_resumen.csv` |
| `06_shap_global.py` | SHAP de S3 por variable original y factores exactos de S1; importancia global | `data/processed/shap_s3.csv`, `shap_s1.csv`, `shap_*_importancia_global.csv` y gráficas |
| `07_shap_segmento.py` | SHAP de S3 para un segmento (`--variable`, `--valor`) | `shap_s3_segmento_*.csv` y gráficas |
| `08_shap_perfil.py` | Cascada de un perfil (S1 por factores y S3 por SHAP), contribuciones y frases | `shap_s1_perfil_*`, `shap_s3_perfil_*` y gráficas |
| `09_frases_lectura.py` | Frases de lectura por plantilla; verifica la nota y que no haya lenguaje causal | `shap_*_frases_prueba.csv`, `shap_verificacion_nota.csv` |
| `10_verificar_aditividad.py` | Base + Σ contribuciones = salida (tolerancia 0.0001) en tres niveles, para S1 y S3 | `shap_aditividad.csv` |

### Fases 4 a 6 (por hacer)

Lanzadores de API y app (`04_app_api/`), evidencia y RAG (`05a_evidencia_rag/`), asistente (`05b_asistente/`) y cierre (`06_cierre/`). Cada uno está descrito en [PLAN.md](PLAN.md). Al agregar un script, súmalo a esta sección.

**Script nuevo:** copia `scripts/_plantilla.py` a `scripts/<fase>/NN_nombre.py` y rellénalo. `run_fase.py` ignora los archivos que empiezan con `_`.

---

## 7. Flujo de trabajo en git

1. Antes de empezar: `git switch main` y `git pull`.
2. Una rama por tarea o fase, con tu nombre: `git switch -c luis/fase-1-datos`.
3. Un commit por tarea terminada, en español e imperativo (`Agrega validación del contrato de datos`). Antes de cada commit revisa `git status` y agrega archivos por nombre, nunca `git add .` a ciegas.
4. Sube tu rama y abre un *pull request*; la otra persona lo revisa y lo integra a `main`. Nadie trabaja directo en `main`.

**Qué no se sube nunca:**
- `data/raw/` (incluido el CSV), `data/processed/` y `data/chroma/`;
- `reports/` y `scratch/`;
- los binarios de `models/`;
- `.venv/` y `.claude/settings.local.json`.

El `.gitignore` ya los excluye.

**Comandos prohibidos** (pueden borrar trabajo o reescribir historial):
- `git reset --hard`;
- cualquier `git push --force`, `-f` o `--force-with-lease`;
- `git clean -fd`;
- `git checkout -- .` y `git restore .` sobre cambios sin guardar;
- `git rebase` de ramas ya subidas y `git filter-branch`.

Claude Code los tiene bloqueados en `.claude/settings.json`. Casi siempre hay una salida segura: `git stash`, `git revert` o una rama nueva.

---

## 8. Convenciones

| Qué | Convención | Ejemplo |
|---|---|---|
| Scripts | `scripts/<NN>_<fase>/<NN>_<nombre>.py`, nombre en español | `scripts/01_datos/02_validacion_contrato.py` |
| Funciones y variables | Inglés, `snake_case`; comentarios y docstrings en español | `load_config()`, `get_evidence()` |
| Ramas | `<persona>/<tarea>` | `luis/fase-3-estrategia-shap` |
| Modelos | `models/<salida>/<salida>-<hash>/` con `modelo.*` y `metadata.json`; `actual.json` apunta a la vigente | `models/s3/s3-e212982330/` |
| Formatos | Tablas en CSV, parámetros en JSON, gráficas en HTML (Plotly); todo en UTF-8 | `reports/01_datos/validacion.csv` |

---

## 9. Solución de problemas

| Problema | Qué hacer |
|---|---|
| `python --version` es menor que 3.10 | Instala Python 3.10 o más nuevo. En Windows, con varias versiones, `py -3.10 -m venv .venv`. |
| PowerShell no activa el entorno ("la ejecución de scripts está deshabilitada") | Ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a `.venv\Scripts\Activate.ps1`. El prompt empieza con `(.venv)` cuando está activo. |
| `ModuleNotFoundError: No module named 'src'` | Corre los scripts desde la raíz del repo (`python scripts/...`), no desde dentro de `scripts/`. |
| `ModuleNotFoundError: No module named 'xgboost'` | El entorno no está activado o no se instaló `lock.txt`. Activa `.venv` e instala. |
| Falta el CSV de datos | Cópialo a `data/raw/base_cards_sintetica.csv` con ese nombre exacto. |
| Un script de la Fase 2 o 3 no encuentra un archivo de `data/processed/` | Corre antes las fases anteriores (`run_fase.py 1`, luego `2`). |
| Ollama apagado (el asistente no responde) | Abre Ollama o corre `ollama serve`. Comprueba con `ollama list` que está `llama3.1:8b`. |
| Poca memoria al cargar el CSV | Cierra otras aplicaciones y no abras el CSV en Excel. Apaga Ollama mientras corres los scripts de datos. |
| `pip install` falla | Revisa que el entorno esté activado y que Python sea 3.10 o superior. Avisa antes de cambiar versiones en `requirements/`. |
