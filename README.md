# Sistema de Inteligencia Crediticia (Tarjetas de Crédito)

## 1. Qué es y en qué estado está

Plataforma **local** que, para un perfil de adquisición de tarjetas de crédito, estima:

| Salida | Pregunta |
|---|---|
| S1. Línea promedio esperada | ¿Qué límite promedio recibe una cuenta de este perfil? |
| S2. Monto total esperado | ¿Cuánto crédito implica colocar N cuentas? (cuentas planeadas × S1) |
| S3. Tasa de impago esperada | ¿Qué fracción de cuentas estará en Over30 a 6 meses? |
| S4. Estrategia sugerida | ¿Qué estrategia rinde mejor para este perfil? |

Cada resultado se explica con SHAP y se puede consultar en lenguaje natural con un asistente (RAG + Ollama) que **solo redacta sobre cifras ya calculadas**. No aprueba ni rechaza créditos, no predice por cliente individual y no afirma causalidad. Los datos del proyecto son una **base sintética**.

**Estado:** infraestructura lista (Fases 0 y 0b) y Fase 1 (datos) terminada, pendiente de revisión: base validada, datasets de S1 y S3 generados y decisiones de datos en [docs/datos_fase1.md](docs/datos_fase1.md). Siguen la Fase 2 (modelos, Luis) y la Fase 5B (agente con ejemplos simulados, Ernesto). El avance detallado está en [PLAN.md](PLAN.md) y las reglas del proyecto en [CLAUDE.md](CLAUDE.md).

---

## 2. Requisitos

| Qué | Para qué | Nota |
|---|---|---|
| **Python 3.10 o superior**, con pip | Todo el proyecto | El código es compatible con 3.10. `requirements/lock.txt` tiene versiones que instalan desde 3.10. |
| **Git** | Trabajar con el repo | |
| **Ollama** con el modelo `llama3.1:8b` | Solo el asistente (Fase 5B) | Es un programa aparte, no un paquete de Python. Se instala desde ollama.com. |

Nada depende de Docker ni de servicios externos: todo corre en tu máquina.

---

## 3. Instalación

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

**Comprobación:** `python -m pytest` debe terminar con todas las pruebas en verde (`passed`). Después copia el CSV de datos (ver §5).

**Ollama (solo para el asistente):** instala Ollama, luego `ollama pull llama3.1:8b` y comprueba con `ollama list`. El servidor local queda en `http://localhost:11434` (configurable en `configs/config.json`, sección `ollama`).

---

## 4. Estructura del repo

```
├── CLAUDE.md  PLAN.md  README.md  pytest.ini
├── .claude/settings.json          # bloqueo de comandos de git destructivos para Claude Code
├── configs/config.json            # semilla, horizonte, particiones, rutas, evidence_mode, Ollama
├── requirements/                  # base.txt, ml.txt, lock.txt
├── data/
│   ├── raw/                       # base_cards_sintetica.csv (NO va a git)
│   ├── contracts/                 # diccionario de variables y reglas del contrato de datos
│   ├── processed/                 # tablas derivadas en CSV (NO va a git)
│   ├── mock/                      # ejemplos simulados de evidencia (evidencia_c1.json …)
│   ├── knowledge/                 # documentos base del RAG, en markdown
│   ├── eval/                      # preguntas de evaluación del asistente
│   └── chroma/                    # índice vectorial (se crea al indexar; NO va a git)
├── scripts/                       # pasos ejecutables, numerados
│   ├── run_fase.py  _plantilla.py
│   ├── 01_datos/  02_modelos_prediccion/  03_estrategia_y_explicabilidad/
│   └── 04_app_api/  05a_evidencia_rag/  05b_asistente/  06_cierre/
├── src/                           # funciones reutilizables (las importan scripts, API, app y agente)
│   ├── utils/  data/  features/  models/  strategy/  consistency/
│   └── explain/  evidence/  rag/  assistant/
├── models/                        # <salida>/<versión>/{modelo.*, metadata.json} y actual.json
├── reports/                       # tablas y gráficas que generan los scripts (NO va a git)
├── api/  app/                     # FastAPI y Streamlit (Fase 4)
├── docs/                          # propuesta, contratos/, fichas de modelo, manuales
├── scratch/                       # pruebas personales: scratch/luis/, scratch/ernesto/ (NO va a git)
└── tests/                         # pruebas con pytest
```

| Carpeta | Qué contiene | Quién la mantiene | ¿Va a git? |
|---|---|---|---|
| `configs/` | `config.json`: parámetros y rutas | Ambos | Sí |
| `requirements/` | Librerías por capas y versiones fijadas | Ambos | Sí |
| `data/raw/` | CSV original de datos | Cada quien lo copia | **No** |
| `data/contracts/` | Diccionario de variables, mapa de columnas, reglas | Luis | Sí |
| `data/processed/` | Datasets derivados (CSV) | Lo generan los scripts de Luis | **No** |
| `data/mock/` | Ejemplos simulados del paquete de evidencia | Luis (formato acordado con Ernesto) | Sí |
| `data/knowledge/` | Documentos del RAG (markdown) | Luis | Sí |
| `data/eval/` | Preguntas de evaluación del asistente | Ernesto | Sí |
| `data/chroma/` | Índice vectorial de Chroma | Lo genera un script | **No** |
| `scripts/01_datos/`, `02_…`, `03_…`, `05a_…` | Pasos de datos, modelos, estrategia, SHAP, evidencia y RAG | Luis | Sí |
| `scripts/05b_asistente/` | Pasos del agente | Ernesto | Sí |
| `scripts/04_app_api/`, `06_cierre/` | Lanzadores de API y app; cierre | Ambos | Sí |
| `src/data`, `features`, `models`, `strategy`, `consistency`, `explain`, `evidence`, `rag` | Lógica de datos, modelos y evidencia | Luis | Sí |
| `src/assistant/` | Lógica del agente | Ernesto | Sí |
| `src/utils/` | Configuración, semilla y estilo común de gráficas | Ambos | Sí |
| `models/` | Modelos entrenados | Luis | Solo `metadata.json` y `actual.json` |
| `reports/` | Tablas (CSV) y gráficas (HTML) generadas | Los scripts | **No** |
| `api/`, `app/` | Servicios FastAPI e interfaz Streamlit | Ambos | Sí |
| `docs/` | Propuesta, contratos, fichas, manuales | Ambos | Sí |
| `scratch/` | Pruebas personales | Cada quien en su subcarpeta | **No** |
| `tests/` | Pruebas (`pytest`) | Cada quien las de su código; ambos las de contrato | Sí |

---

## 5. Dónde poner cada cosa

| Tipo de archivo | Dónde va | ¿Va a git? |
|---|---|---|
| CSV de datos | `data/raw/base_cards_sintetica.csv` (exactamente ese nombre) | **No** |
| Diccionario de variables | `data/contracts/diccionario_variables.csv` | Sí |
| Documentos del RAG | `data/knowledge/*.md` | Sí |
| Preguntas de evaluación | `data/eval/preguntas_evaluacion.csv` | Sí |
| Ejemplos simulados de evidencia | `data/mock/evidencia_*.json` | Sí |
| Pruebas personales (borradores, experimentos) | `scratch/<tu nombre>/`, por ejemplo `scratch/ernesto/` | **No** |
| Modelos generados | `models/<salida>/<versión>/` (los crean los scripts) | Solo `metadata.json` y `actual.json` |
| Reportes (tablas y gráficas) | `reports/<fase>/` (los crean los scripts) | **No** |
| Modelos de Ollama | Fuera del repo: los guarda Ollama en su propia carpeta | No aplica |

### Qué tiene que hacer cada persona para trabajar con sus datos

**Luis (datos y modelos):**
1. Copia el CSV a `data/raw/base_cards_sintetica.csv`. No lo renombres y no lo subas a git (el `.gitignore` ya lo excluye).
2. Corre los scripts de su fase (`python scripts/run_fase.py 1`); los datasets quedan en `data/processed/` y los reportes en `reports/`.

**Ernesto (agente):**
1. No necesita el CSV para empezar: trabaja con los ejemplos de `data/mock/`, que sirve `get_evidence()` con `evidence_mode = "mock"` (ya es el valor de `configs/config.json`).
2. Instala Ollama y descarga `llama3.1:8b`.
3. Sus preguntas de evaluación van en `data/eval/` y sus pruebas personales en `scratch/ernesto/`.
4. Cuando la evidencia real esté lista (Fase 5A), se cambia `evidence_mode` a `"real"`; en ese momento sí hace falta el CSV en `data/raw/`.

---

## 6. Cómo correr

Siempre desde la **raíz del repo** y con el entorno activado.

| Qué | Comando |
|---|---|
| Un script | `python scripts/01_datos/01_carga_y_diccionario.py` |
| Un script sin gráficas | `python scripts/01_datos/03_exploracion_cohortes.py --sin-graficas` |
| Ayuda de un script | `python scripts/<fase>/<script>.py --help` |
| Una fase completa, en orden | `python scripts/run_fase.py 1` (fases: 1, 2, 3, 4, 5a, 5b, 6) |
| Continuar una fase desde un script | `python scripts/run_fase.py 1 --desde 03` |
| Todas las pruebas | `python -m pytest` |

`run_fase.py` se detiene en el primer error, imprime un resumen (script, resultado, segundos) y dice cómo continuar. Si la fase todavía no tiene scripts, avisa y termina sin error.

**Dónde aparecen las salidas:** datasets en `data/processed/`, tablas (CSV) y gráficas (HTML, se abren en el navegador) en `reports/<fase>/`, modelos en `models/`. Cada script imprime al final qué archivos escribió.

**Nuevo script:** copia `scripts/_plantilla.py` a `scripts/<fase>/NN_nombre.py` y rellénalo.

---

## 7. Tabla de scripts

Los scripts de la Fase 1 ya existen; los demás son los planeados en [PLAN.md](PLAN.md) y la tabla se actualiza al agregar cada uno.

| Script | Qué hace | Lee | Escribe | Quién |
|---|---|---|---|---|
| `scripts/run_fase.py` | Corre en orden los scripts de una fase | `scripts/<fase>/` | Resumen en la terminal | Ambos |
| `scripts/_plantilla.py` | Plantilla para scripts nuevos | — | — | Ambos |
| `01_datos/01_carga_y_diccionario.py` | Carga el CSV (nombres reales o alias `dummiN`), quita espacios a las categorías y verifica `src/data/catalog.py` contra el diccionario | CSV de datos, diccionario | `data/contracts/column_map.json`, `data/processed/base_estandar.csv`, `reports/01_datos/carga_*.csv` | Luis |
| `01_datos/02_validacion_contrato.py` | Valida el contrato de datos (sale con código 1 si hay fallas bloqueantes); comprueba qué columnas son acumuladas | `base_estandar.csv` (o `--entrada`) | `reports/01_datos/validacion.csv`, `columnas_acumuladas.csv`, `poblacion_columnas.csv` | Luis |
| `01_datos/03_exploracion_cohortes.py` | Cuentas por vintage, estrategia y segmento; producto, canal, score y edad; filas por MOB | `base_estandar.csv` | `reports/01_datos/exploracion_*.csv` y `graficas/exploracion_*.html` | Luis |
| `01_datos/04_madurez_y_curvas.py` | Madurez a 6 meses, curvas de Over30, `OVER30ACCTS > ENRACCTS` en MOB 6 y datasets de modelado | `base_estandar.csv` | `data/processed/s1_dataset.csv`, `s3_dataset.csv`; `reports/01_datos/madurez_vintages.csv`, `curvas_*.csv`, `over30_mayor_enr_mob6.csv` y gráficas | Luis |
| `01_datos/05_balance_entre_estrategias.py` | Balance de perfiles r.Reca champ vs. DARIA 1.6 en 202511–202607, asignación por vintage y estrategias evaluables | `base_estandar.csv` | `reports/01_datos/balance_estrategias.csv`, `balance_asignacion_por_vintage.csv`, `estrategias_evaluables.csv` y gráficas | Luis |
| `02_modelos_prediccion/00` a `06` | Variables, partición, líneas base y XGBoost de S1 y S3, S2, resumen | Datasets de modelado | `models/s1/`, `models/s3/`, `reports/`, `docs/fichas/` | Luis |
| `03_estrategia_y_explicabilidad/01` a `10` | S4, reglas de consistencia y SHAP | Modelos y datasets | `reports/`, `src/strategy/`, `src/consistency/`, `src/explain/` | Luis |
| `04_app_api/run_api.py`, `run_app.py` | Lanzan la API y la interfaz | Modelos | — | Ambos |
| `05a_evidencia_rag/01` a `07` | Base documental, índice Chroma y evidencia C1–C7 | `data/knowledge/`, modelos | `data/chroma/`, `src/evidence/` | Luis |
| `05b_asistente/01` a `07` | Ollama, clasificador, redactor, barrera, chat y evaluación | `data/mock/`, `data/eval/` | `src/assistant/`, `reports/05b_asistente/` | Ernesto |
| `06_cierre/01`, `02` | Pruebas de extremo a extremo y criterios de aceptación | Todo lo anterior | Informe de criterios | Ambos |

---

## 8. Flujo de trabajo en git

1. Antes de empezar: `git switch main` y `git pull`.
2. Crea una rama por tarea o fase, con tu nombre: `git switch -c luis/fase-1-datos`.
3. Un commit por tarea terminada. Antes de cada commit revisa `git status` y agrega archivos por nombre (no `git add .` a ciegas).
4. Sube tu rama y abre un *pull request*. **La otra persona lo revisa** y lo integra a `main`. Nadie trabaja directo en `main`.

**Qué no se sube nunca:** el CSV de datos y todo `data/raw/` y `data/processed/`, `data/chroma/`, `reports/`, `scratch/`, los binarios de `models/`, `.venv/` y `.claude/settings.local.json`. El `.gitignore` ya los excluye.

**Comandos prohibidos** (pueden borrar trabajo o reescribir historial): `git reset --hard`, cualquier `git push --force` / `-f` / `--force-with-lease`, `git clean -fd`, `git checkout -- .` y `git restore .` sobre cambios sin guardar, `git rebase` de ramas ya subidas y `git filter-branch`. Claude Code los tiene bloqueados en `.claude/settings.json`. Si crees que necesitas uno, habla primero con la otra persona: casi siempre hay una salida segura (`git stash`, `git revert`, una rama nueva).

---

## 9. Convenciones de nombres

| Qué | Convención | Ejemplo |
|---|---|---|
| Scripts | `scripts/<NN>_<fase>/<NN>_<nombre>.py`, número de dos dígitos y nombre en español | `scripts/01_datos/02_validacion_contrato.py` |
| Funciones y variables | Inglés, `snake_case`; comentarios y docstrings en español | `load_config()`, `get_evidence()` |
| Ramas | `<persona>/<tarea>` | `luis/fase-1-datos`, `ernesto/asistente-clasificador` |
| Commits | Español, imperativo, cortos | `Agrega validación del contrato de datos` |
| Modelos | `models/<salida>/<versión>/` con `modelo.*` y `metadata.json`; `actual.json` apunta a la versión vigente | `models/s1/v001/metadata.json` |
| Salidas | Tablas en CSV, parámetros en JSON y gráficas en HTML | `reports/01_datos/validacion.csv` |

---

## 10. Reparto y contratos

| Persona | Frente |
|---|---|
| **Luis** | Datos, modelos (S1, S2, S3, S4), consistencia, SHAP, evidencia determinista y RAG |
| **Ernesto** | El agente: clasificador de intención, redacción, barrera numérica, flujo del chat y su evaluación |
| **Ambos** | App y API, contratos, documentación y configuración |

Los dos frentes se comunican **solo por contratos**, en `docs/contratos/`:

- [evidencia.md](docs/contratos/evidencia.md): catálogo de consultas C1–C7 y formato del paquete de evidencia, con un ejemplo por consulta.
- [chat.md](docs/contratos/chat.md): entrada y salida del servicio de chat.

Cambiar un contrato requiere el acuerdo de ambos. Mientras no exista la evidencia real, `get_evidence()` (en `src/evidence/`) devuelve los ejemplos de `data/mock/`; las pruebas de `tests/test_evidence_contract.py` verifican que cumplen el contrato.

---

## 11. Solución de problemas

| Problema | Qué hacer |
|---|---|
| **Versión de Python**: `python --version` no es 3.10 o superior | Instala Python 3.10 o más nuevo. En Windows, con varias versiones instaladas, puedes elegir una con `py -3.10 -m venv .venv`. |
| **No se activa el entorno** en PowerShell ("la ejecución de scripts está deshabilitada") | Ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a `.venv\Scripts\Activate.ps1`. Para saber si está activo, el prompt empieza con `(.venv)`. |
| **`ModuleNotFoundError: No module named 'src'`** | Corre los scripts desde la raíz del repo (`python scripts/...`), no desde dentro de `scripts/`. |
| **Falta el CSV de datos** | Copia el archivo a `data/raw/base_cards_sintetica.csv` con ese nombre exacto. No viene en git; pídeselo a Luis. |
| **Ollama apagado** (el asistente no responde o da error de conexión) | Abre la aplicación de Ollama o corre `ollama serve`. Comprueba con `ollama list` que aparece `llama3.1:8b`; si no, `ollama pull llama3.1:8b`. |
| **Poca memoria** (el equipo se pone lento o el script se cierra al cargar el CSV) | Cierra otras aplicaciones. No abras el CSV completo en Excel. Ollama con `llama3.1:8b` también usa bastante memoria: apágalo mientras corres los scripts de datos. |
| **`pip install` falla con una librería** | Revisa que el entorno esté activado y que la versión de Python sea 3.10 o superior; avisa en el equipo antes de cambiar versiones en `requirements/`. |
