# PLAN.md — Plan de trabajo

> **Cómo se usa:** cada tarea es un script (o una pieza de `src/`). Al terminar una, marca su casilla `[x]`.
> No pases a la siguiente fase sin mi visto bueno. Las reglas y convenciones están en `CLAUDE.md`; la guía de uso del repo, en `README.md`.
> **Responsables:** `[L]` Luis · `[E]` Ernesto · `[L+E]` ambos. El avance se mide por tareas terminadas.

---

## Vista general

| Fase | Qué se hace | Quién | Estado |
|---|---|---|---|
| 0. Preparación | Repo, entorno, configuración base | L | ✅ |
| **0b. Reestructura** | Scripts en vez de notebooks, README completo, bloqueo técnico de git, contratos y ejemplos simulados | L | ✅ |
| 1. Datos | Cargar, validar y entender el dataset | L | ⬜ |
| 2. Modelos S1, S2 y S3 | Líneas base y XGBoost, métricas, versionado | L | ⬜ |
| 3. Estrategia, consistencia y SHAP | S4, reglas de consistencia, explicabilidad en tres niveles | L | ⬜ |
| 4. App y API | FastAPI + Streamlit | L+E | ⬜ |
| 5A. Evidencia y RAG | Funciones deterministas C1–C7 y base documental | L | ⬜ |
| 5B. Agente | Clasificador, redacción, barrera numérica, flujo del chat, evaluación | E | ⬜ (arranca al terminar la 0b, con ejemplos simulados) |
| 6. Cierre | Pruebas de extremo a extremo, documentación, criterios de aceptación | L+E | ⬜ |

**Qué depende de qué**

```
0b ──► 1 ──► 2 ──► 3 ──► 4 ──► 6
 │                 └──► 5A ──┘   ▲
 └──────────► 5B (con mocks) ────┘
```

- **Primer corte:** fases 0b, 1 y 2 (Luis) y el arranque de 5B (Ernesto).
- Ernesto **no espera** a Luis: trabaja con los ejemplos de `data/mock/`. Cuando 5A entrega evidencia real, se cambia `evidence_mode` de `"mock"` a `"real"` en `configs/config.json`.

---

## Contrato Luis ↔ Ernesto (borrador; lo afinan los dos en la Fase 0b)

Los dos frentes solo se hablan por estos contratos (`docs/contratos/`). Cambiarlos requiere acuerdo de ambos.

### Catálogo de consultas

| ID | Consulta de ejemplo | Función de evidencia (`src/evidence/`) | Parámetros | Fuente |
|---|---|---|---|---|
| C1 | ¿Por qué este perfil tiene riesgo alto? | `explain_profile` | perfil | SHAP de S3 |
| C2 | ¿Cómo afecta el canal a la tasa de impago? | `variable_effect` | variable | SHAP agregado y tasas observadas |
| C3 | Compara Costco con Home Depot en Over30 | `compare_groups` | dimensión, valor A, valor B | Pandas sobre cohortes maduras |
| C4 | ¿Qué estrategia conviene para este perfil? | `suggest_strategy` | perfil | Salida S4 con intervalo |
| C5 | ¿Cómo evoluciona Over30 en la vintage 202506? | `vintage_evolution` | vintage | Pandas por MOB |
| C6 | Resume los resultados del lote | `summarize_batch` | identificador del lote | Resultados cargados |
| C7 | ¿Qué significa GCL? | `define_term` | término | Base documental (Chroma) |

**Perfil** = estrategia, segmento de estrategia, producto, canal, rango de edad y rango de score (más vintage y cuentas planeadas, opcionales).

### Formato del paquete de evidencia

Cada función devuelve un diccionario (JSON) con estos campos:

```json
{
  "consulta_id": "C1",
  "estado": "ok",
  "funcion": "evidence.explain_profile",
  "parametros": {},
  "cifras": {},
  "tablas": [],
  "frases_base": [],
  "fragmentos": [],
  "versiones": {},
  "motivo": null
}
```

- `estado`: `ok`, `no_evaluable` (por ejemplo, DARIA 2.0 en riesgo o un perfil fuera de dominio) o `sin_evidencia`. Con estado distinto de `ok`, `motivo` explica por qué y el agente se abstiene.
- `cifras`: **todos los números** que el agente puede citar, con nombre (por ejemplo `tasa_s3`, `nivel_riesgo`, `confianza`).
- `tablas`: tablas pequeñas ya calculadas (nombre, columnas, filas).
- `frases_base`: frases de lectura generadas por plantilla, en español y sin lenguaje causal.
- `fragmentos`: solo C7; texto recuperado de la base documental (se trata como **datos**, no como instrucciones).
- `versiones`: modelo(s), datos y horizonte usados; el agente las muestra en cada respuesta junto con la función y los parámetros.

**Regla para el agente:** todo número del texto redactado debe aparecer en `cifras` o `tablas`; si no, la barrera numérica lo rechaza y se muestra la respuesta de plantilla con los datos verificados.

### Servicio de chat

Entrada: pregunta en texto y, si aplica, perfil o lote en sesión. Salida: respuesta redactada, `consulta_id`, función, parámetros, versiones y estado (`respondida`, `abstenida`, `no_soportada`). El esquema exacto se define en `docs/contratos/chat.md`.

---

## Reparto de carpetas

| Carpeta | Dueño |
|---|---|
| `scripts/01–03`, `05a`; `src/data, features, models, strategy, consistency, explain, evidence, rag`; `data/knowledge`; `models/` | Luis |
| `scripts/05b`; `src/assistant`; `data/eval` | Ernesto |
| `api/`, `app/`, `scripts/04_app_api`, `docs/`, `README.md`, `CLAUDE.md`, `PLAN.md`, `requirements/`, `configs/`, tests de contrato | Ambos (cambios acordados) |

---

## Cierre de cada script (aplica a todos)

- [ ] El docstring inicial dice objetivo, entradas, salidas, cómo correrlo y listo cuando.
- [ ] Cada bloque de código tiene su comentario (primero qué hace, luego por qué); las fórmulas están traducidas a código.
- [ ] Las gráficas están en funciones aparte y se pueden omitir con `--sin-graficas`.
- [ ] Corre completo desde la raíz, sin intervención, y repetirlo da el mismo resultado.
- [ ] La lógica reutilizable ya está en `src/` y tiene prueba en `tests/`.
- [ ] `README.md` actualizado si se agregó un script o una carpeta.
- [ ] Casilla marcada en este archivo y commit hecho.

---

## Fase 0 — Preparación ✅

- [x] Estructura de carpetas, `.gitignore`, entorno virtual, `requirements/base.txt`, `ml.txt` y `lock.txt`.
- [x] `configs/config.json` (semilla 42, H = 6, particiones, rutas), `src/utils/` (`load_config`, `get_path`, `set_seed`) y sus pruebas.
- [x] Datos en `data/raw/`, diccionario en `data/contracts/`, propuesta en `docs/`, `README.md` inicial y primer commit.

---

## Fase 0b — Reestructura `[L]` ✅

**Alcance:** solo infraestructura y contratos. No cargues ni analices datos todavía. Trabaja en la rama `luis/fase-0b-reestructura`; el merge a `main` lo hago yo (Ernesto necesita los contratos y los ejemplos).

- [x] **De notebooks a scripts.** Con `git mv`, reemplaza `notebooks/` por `scripts/` con las subcarpetas: `01_datos`, `02_modelos_prediccion`, `03_estrategia_y_explicabilidad`, `04_app_api`, `05a_evidencia_rag`, `05b_asistente`, `06_cierre`. En el repo no debe quedar ningún archivo `.ipynb`, y el `README.md` y `docs/` no deben mencionar notebooks.
- [x] **Carpetas nuevas** (con `.gitkeep`): `reports/`, `data/mock/`, `data/knowledge/`, `data/eval/`, `scratch/`, `docs/contratos/`, `src/strategy/`, `src/evidence/`, `src/rag/`.
- [x] **`.gitignore`:** agrega `reports/`, `scratch/`, `data/chroma/`, `.claude/settings.local.json`; en `models/` ignora `models/**/modelo.*` pero **sí** versiona `metadata.json` y `actual.json`. Verifica con `git status` y `git check-ignore -v`.
- [x] **`scripts/_plantilla.py`:** plantilla con el docstring estándar, el encabezado para importar `src`, secciones numeradas, `main()`, gráficas en funciones aparte y `argparse` con `--sin-graficas` (ver `CLAUDE.md` §9).
- [x] **`scripts/run_fase.py`:** `python scripts/run_fase.py 1` corre en orden los scripts de la fase; se detiene en el primer error e imprime un resumen. Opción `--desde NN` para continuar desde un script.
- [x] **`configs/config.json`:** agrega las carpetas nuevas, `evidence_mode` (`"mock"`), `ollama` (servidor local, modelo `llama3.1:8b`, tiempo máximo de espera) y la ruta de Chroma.
- [x] **Bloqueo técnico de git en `.claude/settings.json`** (se versiona, para que aplique también en la máquina de Ernesto): reglas de denegación para `git reset --hard`, `git push --force`, `git push -f`, `git push --force-with-lease` (con el flag antes y después de los argumentos) y `git clean -fd`. Usa la sintaxis de permisos que acepte la versión de Claude Code instalada y **compruébalo** (por ejemplo con `/permissions`). Dime qué variantes quedaron cubiertas y cuáles no.
- [x] **`docs/contratos/evidencia.md`:** catálogo C1–C7 y formato del paquete de evidencia, tal como están en este archivo, con la descripción de cada campo y un ejemplo completo por consulta.
- [x] **`docs/contratos/chat.md`:** entrada y salida del servicio de chat (campos, estados, ejemplo).
- [x] **Ejemplos simulados en `data/mock/`:** `evidencia_c1.json` … `evidencia_c7.json`, más uno con `estado = "no_evaluable"` y otro con `estado = "sin_evidencia"`. Las cifras son inventadas a propósito y llevan una marca `"simulado": true`.
- [x] **`src/evidence/`:** función `get_evidence(consulta_id, **params)` que, con `evidence_mode = "mock"`, devuelve el ejemplo correspondiente; con `"real"` lanza `NotImplementedError` con un mensaje claro. Pruebas en `tests/` que validan que cada ejemplo cumple el contrato.
- [x] **`README.md` completo** con estas secciones:
  1. Qué es el proyecto y en qué estado está.
  2. Requisitos (Python, Git y, para el agente, Ollama).
  3. Instalación paso a paso en Windows (PowerShell, incluida la política de ejecución para activar el entorno) y en macOS/Linux, y cómo comprobar que todo funciona con `pytest`.
  4. **Estructura del repo:** el árbol completo y una tabla carpeta por carpeta (qué contiene, quién la mantiene, si va a git).
  5. **Dónde poner cada cosa:** tabla por tipo de archivo (CSV de datos, diccionario, documentos del RAG, preguntas de evaluación, ejemplos simulados, pruebas personales en `scratch/<nombre>/`, modelos generados, reportes, modelos de Ollama, que viven fuera del repo) y **qué debe hacer cada persona para trabajar con sus datos** (Luis y Ernesto).
  6. Cómo correr: un script, una fase completa, las pruebas; dónde aparecen las salidas.
  7. Tabla de scripts: nombre, qué hace, qué lee, qué escribe, quién lo mantiene.
  8. Flujo de trabajo en git: ramas, *pull requests* revisados por la otra persona, qué no se sube, comandos prohibidos.
  9. Convenciones de nombres (scripts, ramas, commits).
  10. Reparto Luis/Ernesto y contratos.
  11. Solución de problemas (versión de Python, activación del entorno, falta el CSV, Ollama apagado, poca memoria).
- [x] Revisar `README.md` y `docs/`: sin fechas ni duraciones de calendario y sin menciones a notebooks.
- [x] Commit(s) en la rama. **No hagas push.**

**Listo cuando:** `pytest` pasa, `python scripts/run_fase.py 1` corre sin error aunque aún no haya scripts (avisa que la fase está vacía), `.claude/settings.json` bloquea los comandos prohibidos, y alguien que no conoce el repo puede ubicar dónde poner sus datos solo con el README.

---

## Fase 1 — Datos `[L]` (`scripts/01_datos/`) ✅

Objetivo: cohortes limpias, validadas y entendidas, con el contrato de datos aprobado.

- [x] **`01_carga_y_diccionario.py`**
  - Hace: carga `data/raw/base_cards_sintetica.csv` en UTF-8 con tipos económicos (categóricos donde convenga); imprime forma, tipos, nulos y memoria; detecta si las columnas traen alias `dummiN` o nombres reales y renombra a los reales; lee el diccionario y construye `src/data/catalog.py` (catálogos de producto, canal, edad, segmentos y scores por familia, estrategia según vintage, rangos mínimo y máximo); revisa espacios al final en las categorías.
  - Entra → sale: CSV de datos y diccionario → `data/contracts/column_map.json`, `src/data/catalog.py`, `src/data/load.py`, `data/processed/base_estandar.csv`.
  - Listo cuando: el CSV se carga con nombres estándar, queda documentado cuál formato traía y qué hacer con `MOB` si no tiene alias, y `catalog.py` tiene pruebas.
- [x] **`02_validacion_contrato.py`**
  - Hace: implementa `src/data/validation.py` con pandas puro (reglas bloqueantes y avisos de `CLAUDE.md` §5); comprueba con evidencia cuáles columnas de resultado son acumuladas (`OVER30ACCTS` y `REVENUE` ya están confirmadas) y a qué población se refieren `REVENUE` y `EXPENSES_TOTAL`.
  - Entra → sale: `data/processed/base_estandar.csv` → `reports/01_datos/validacion.csv` (regla, fila, qué falló, qué hacer) y `reports/01_datos/columnas_acumuladas.csv`.
  - Listo cuando: el validador pasa con el dataset real, falla con casos rotos a propósito (pruebas en `tests/`), sale con error ante fallas bloqueantes y las decisiones quedan documentadas.
- [x] **`03_exploracion_cohortes.py`**
  - Hace: cuentas por vintage, estrategia y segmento; participación de productos y canales; distribución de score y edad por familia; filas por MOB.
  - Entra → sale: base estándar → tablas en `reports/01_datos/` y gráficas HTML.
  - Listo cuando: las tablas y gráficas se generan con una sola corrida y yo las reviso.
- [x] **`04_madurez_y_curvas.py`**
  - Hace: función de madurez a H = 6 (`src/data/cohorts.py`), con comprobación de las 27 vintages (202401–202603); curvas de la tasa de Over30 por MOB por vintage y estrategia; **proporción de cohortes con `OVER30ACCTS > ENRACCTS` en MOB = 6**; construye los conjuntos de modelado.
  - Entra → sale: base estándar → `data/processed/s1_dataset.csv` (línea en MOB = 0, todas las vintages), `data/processed/s3_dataset.csv` (tasa en MOB = 6, solo cohortes maduras) y reportes.
  - Listo cuando: ambos datasets están generados y, si la proporción de `OVER30ACCTS > ENRACCTS` es alta, queda una propuesta escrita sobre el denominador de S3 para que yo decida.
- [x] **`05_balance_entre_estrategias.py`**
  - Hace: aísla la ventana de asignación aleatoria (202511–202607) y compara la mezcla de perfiles entre r.Reca champ y DARIA 1.6 (edad, producto, canal, posición de score); marca DARIA 2.0 como *no evaluable en riesgo*.
  - Entra → sale: base estándar → `reports/01_datos/balance_estrategias.csv` y gráficas.
  - Listo cuando: el diagnóstico de balance queda listo para que lo use S4 en la Fase 3.

**Listo cuando (fase):** el validador bloquea los archivos inválidos con errores legibles, los dos datasets están generados y yo los revisé.

---

## Fase 2 — Modelos S1, S2 y S3 `[L]` (`scripts/02_modelos_prediccion/`) ✅

Objetivo: S1 y S3 con línea base y XGBoost, comparados en validación y prueba, versionados.

- [x] **`00_variables_y_particion.py`**
  - Hace: `src/features/engineering.py` con one-hot (estrategia, segmento, producto, canal), edad ordinal, **posición relativa del score 0–1 dentro de su familia** más la familia, y mes de booking derivado de la vintage; partición temporal por vintage (entrenamiento 202401–202508, validación 202509–202511, prueba 202512–202603; S1 hasta 202609); **prueba automática de que no hay fuga** (ninguna vintage en dos conjuntos; ninguna columna de resultado, MOB, vintage numérica ni pesos entre las variables).
  - Entra → sale: `s1_dataset.csv`, `s3_dataset.csv` → matrices y particiones en `data/processed/`.
  - Listo cuando: la prueba de fuga pasa y el conteo de filas por conjunto coincide con la tabla de `CLAUDE.md` §7.
- [x] **`01_s1_linea_base.py`**
  - Hace: media ponderada por producto y score; GLM Gamma (scikit-learn) con peso por cuentas; métricas en validación y prueba (MAE, MAPE, R² y sesgo ponderados).
  - Entra → sale: matrices de S1 → métricas en `reports/02_modelos_prediccion/`.
  - Listo cuando: las métricas de ambas líneas base quedan guardadas.
- [x] **`02_s1_xgboost.py`** — el XGBoost no cumplió el criterio (MAE de prueba 2.9% peor que la base simple); por decisión de Luis se publica la **línea base simple** como S1 (ver `docs/fichas/ficha_s1.md`).
  - Hace: XGBoost sobre el logaritmo de la línea, peso = `ACCOUNTS`, árboles poco profundos; Optuna (50 pruebas) con parada temprana (50 rondas) y semilla 42, sobre validación; mismas métricas contra la línea base y reporte por estrategia; guarda modelo y `metadata.json` (`src/models/registry.py`).
  - Entra → sale: matrices de S1 → `models/s1/<versión>/` y `models/s1/actual.json`.
  - Listo cuando: **el MAE de prueba es al menos 10% menor que el de la línea base simple**; si no, no se publica y me avisas.
- [x] **`03_s2_monto_total.py`**
  - Hace: función `S2 = cuentas planeadas × S1` en `src/`, con prueba (resultado exacto).
  - Entra → sale: S1 y cuentas planeadas → monto total.
  - Listo cuando: la prueba confirma que S2 es exactamente cuentas planeadas por S1.
- [x] **`04_s3_linea_base.py`**
  - Hace: tasa ponderada por score y producto; GLM binomial (cada fila aporta tantos eventos como cuentas en Over30 y tantos no eventos como cuentas expuestas restantes); AUC, KS, Gini, log loss y Brier ponderados; calibración por deciles.
  - Entra → sale: matrices de S3 → métricas en `reports/`.
  - Listo cuando: las métricas de ambas líneas base quedan guardadas.
- [x] **`05_s3_xgboost.py`**
  - Hace: XGBoost con etiqueta fraccional, peso = `ENRACCTS`, restricción monótona decreciente en el score; Optuna y parada temprana como en S1; calibración ajustada **solo en validación** (por ejemplo, regresión isotónica o de Platt con scikit-learn; documenta cuál y por qué); métricas contra la línea base, calibración por deciles y reporte por estrategia y por rango de edad; guarda modelo y `metadata.json`.
  - Entra → sale: matrices de S3 → `models/s3/<versión>/`, `models/s3/actual.json`.
  - Listo cuando: **no es peor que el GLM, la razón observado/predicho está entre 0.95 y 1.05 y la tasa observada crece con el nivel de riesgo**; si no, no se publica y me avisas.
- [x] **`06_resumen_modelos.py`**
  - Hace: tabla resumen de S1 y S3 (modelo contra línea base, validación contra prueba) y ficha corta de cada modelo en `docs/fichas/`. Deja constancia de que el conjunto de prueba se evaluó **una sola vez** y de que la base es sintética.
  - Entra → sale: metadatos y métricas → `docs/fichas/ficha_s1.md`, `ficha_s3.md`.
  - Listo cuando: las fichas están completas y yo las revisé.

**Listo cuando (fase):** ambos XGBoost superan a su línea base según los criterios, están versionados con su metadata y las fichas están revisadas.

---

## Fase 3 — Estrategia, consistencia y SHAP `[L]` (`scripts/03_estrategia_y_explicabilidad/`)

Objetivo: S4 con intervalos, reglas de consistencia y explicaciones en tres niveles.

**3A. Estrategia (S4)**

- [x] **`01_ventana_aleatoria.py`**
  - Hace: toma solo la ventana con asignación aleatoria (202511–202607), usa solo cohortes maduras a H = 6 y arma una tabla por perfil con r.Reca champ y DARIA 1.6; excluye DARIA 2.0. Compara solo en el **soporte común** (perfiles con cuentas en ambas estrategias; por ejemplo, DARIA 1.6 no tiene Costco ni Beyond); los perfiles fuera de él salen como "sin evidencia".
  - Entra → sale: `s3_dataset.csv` y el balance de la Fase 1 → `data/processed/s4_ventana.csv`.
  - Listo cuando: el balance de perfiles entre estrategias queda verificado y documentado.
- [x] **`02_s4_observado_bootstrap.py`**
  - Hace: método 1. Diferencia observada de la tasa de Over30 entre estrategias, con intervalo por *bootstrap* sobre cohortes (semilla 42).
  - Entra → sale: `s4_ventana.csv` → diferencias e intervalos por perfil en `reports/03_.../`.
  - Listo cuando: cada perfil tiene diferencia e intervalo, y el *bootstrap* es reproducible.
- [x] **`03_s4_prediccion_escenarios.py`**
  - Hace: método 2. Predice con S3 el mismo perfil bajo cada estrategia y calcula la diferencia (con intervalo).
    *Decisión:* se predicen las cohortes reales de cada estrategia y se promedian ponderando por ENRACCTS, con un S3 reentrenado **sin la vintage que se predice** (5 vintages en la ventana, mismos hiperparámetros, rondas y calibración del publicado), para que la concordancia no salga inflada. El intervalo bootstrap del método 2 refleja solo la variación entre cohortes; la decisión usa el del método 1.
  - Entra → sale: modelo S3 y perfiles → diferencias predichas.
  - Listo cuando: cada perfil tiene su diferencia predicha bajo ambas estrategias.
- [x] **`04_s4_sugerencia_y_validacion.py`**
  - Hace: combina ambos métodos con la regla de decisión: se sugiere una estrategia **solo si el intervalo excluye cero y los dos métodos coinciden en el signo**; si no, "Sin diferencia concluyente". DARIA 2.0 sale como *no evaluable en riesgo*. Calcula la concordancia entre métodos y guarda la función en `src/strategy/`.
  - Entra → sale: salidas de los dos métodos → `reports/` con la sugerencia por perfil.
  - Listo cuando: **el signo coincide en al menos 70% de los perfiles con diferencia concluyente**, con pruebas de la regla de decisión.

**3B. Consistencia**

- [x] **`05_reglas_consistencia.py`**
  - Hace: módulo determinista (`src/consistency/`) que corre después de los modelos: calcula S2 en lugar de predecirlo; recorta las tasas al rango 0–1; marca como *no evaluable* un perfil fuera del dominio de entrenamiento; baja la confianza cuando el soporte histórico es escaso (Alta con 1,000 o más cuentas maduras en perfiles similares, Media de 200 a 999, Baja con menos de 200); emite alerta cuando el riesgo es Crítico; agrega a cada fila las versiones de modelo y de datos. Calcula los **umbrales de riesgo** (percentiles 50, 80 y 95 de la tasa histórica ponderada) y los guarda en el `metadata.json` de S3.
  - Entra → sale: modelos y datasets → `src/consistency/`, umbrales en `metadata.json`.
  - Listo cuando: **las reglas se cumplen en el 100% de las filas de prueba** y hay pruebas por regla. Antes de implementar, **propónme por escrito** cómo defines "perfil fuera del dominio" y "perfiles similares" y espera mi respuesta.
    *Decisión:* fuera del dominio = la combinación (familia, segmento, producto, canal) no aparece en el entrenamiento del modelo; perfiles similares = misma familia, segmento, producto, canal y edad, con cualquier score. Los umbrales de riesgo se conservan de la Fase 2 (tasa **predicha** en entrenamiento, no la histórica observada): desviación aprobada y documentada en `docs/consistencia_fase3.md`.

**3C. Explicabilidad (SHAP)**

- [ ] **`06_shap_global.py`** *(S3 listo; S1 pendiente de definir su explicación, ver nota abajo)* — Hace: `TreeExplainer` sobre S3 y S1; **suma los valores de las columnas one-hot por variable original**; barras de importancia global en Plotly. Entra → sale: modelos y matrices → valores SHAP en `data/processed/` y gráficas en `reports/`. Listo cuando: las barras global de S1 y de S3 están generadas. En S3, SHAP está en escala logit (se muestra dirección y tamaño relativo, no puntos porcentuales); en S1 se lee como factor multiplicativo.
- [x] **`07_shap_segmento.py`** — Hace: barras y *beeswarm* (con Plotly, a partir de los valores SHAP) filtrados por un segmento (por ejemplo, un canal). Listo cuando: se puede pedir cualquier segmento por parámetro.
- [ ] **`08_shap_perfil.py`** *(S3 listo; S1 pendiente)* — Hace: gráfica de cascada de un perfil, con Plotly (`go.Waterfall`). Listo cuando: un perfil de ejemplo produce su cascada y su tabla de contribuciones.
- [x] **`09_frases_lectura.py`** — Hace: por plantilla (sin modelo de lenguaje), los **tres factores que más suben y los tres que más bajan** la tasa, en español, y la nota fija: *"Estas explicaciones describen cómo usa las variables el modelo; no establecen relaciones causales."* Entra → sale: SHAP por perfil → frases en `src/explain/`. Listo cuando: toda vista con SHAP lleva la nota y las frases no usan lenguaje causal.
- [ ] **`10_verificar_aditividad.py`** *(S3 pasa; S1 pendiente)* — Hace: comprueba que **valor base + suma de contribuciones SHAP = salida del modelo** con tolerancia de 0.0001, para S1 y S3, en los tres niveles. Compara con la salida **antes de la calibración** de S3 y deja escrito que SHAP explica el modelo sin calibrar. Listo cuando: la prueba pasa para cada modelo.

> **Nota SHAP:** se usa el TreeSHAP exacto de XGBoost (`pred_contribs=True`) en lugar de `shap.TreeExplainer`, porque shap 0.49.1 no lee el formato de `base_score` de XGBoost 3.x. Es el mismo algoritmo y la aditividad se verifica en `10_verificar_aditividad.py`. El S1 publicado es la línea base simple (no un árbol): su explicación se define aparte.

**Listo cuando (fase):** criterios de S4, consistencia y SHAP cumplidos y revisados por mí.

---

## Fase 4 — App y API `[L+E]` (`api/`, `app/`, `scripts/04_app_api/`)

Objetivo: una interfaz usable por personal sin formación técnica, con servicios locales. **Antes de empezar, decidamos entre Luis y Ernesto quién hace qué** (propuesta: Luis, servicios de validar, predecir, explicar y comparar; Ernesto, el servicio de chat y su pantalla).

- [ ] **4.1 Contratos de la API (`api/schemas.py`, Pydantic):** entrada y salida de cada servicio. **Entrada de perfiles:** obligatorias estrategia, segmento de estrategia, producto, canal, rango de edad y rango de score; opcionales un identificador de perfil, la vintage (solo para derivar el mes de booking) y las cuentas planeadas (habilitan el monto total). **Entrada histórica:** el esquema del diccionario, para comparar lo observado contra lo predicho. **Salida:** atributos del perfil, línea promedio, monto total, tasa de impago esperada, nivel de riesgo, estrategia sugerida con su intervalo, confianza, estado con su motivo, y versiones de modelo y de datos con el horizonte.
- [ ] **4.2 API simulada (`api/main.py`):** devuelve el formato final con datos de ejemplo, para que la app y el agente no esperen a los modelos.
- [ ] **4.3 Servicios reales (FastAPI, solo localhost):** validar, predecir, explicar, comparar estrategias, chat y estado del sistema. Validación con las reglas de `src/data/validation.py`; los errores indican qué falló, en qué fila y qué hacer. Archivos admitidos de hasta **50 MB en UTF-8**; las columnas no declaradas se descartan. **Nota:** la base histórica completa pesa unos 193 MB, así que decide con Luis cómo se maneja el modo histórico (por ejemplo, muestras o agregados).
- [ ] **4.4 Interfaz (Streamlit):** carga de CSV, tablero, gráficas (Plotly), comparador de estrategias, chat y exportación a CSV con las columnas de salida de 4.1.
- [ ] **4.5 Usabilidad:** cada término técnico (Over30, MOB, vintage, SHAP) tiene una definición breve al pasar el cursor; cada resultado trae una frase de lectura en español generada por plantilla; los niveles de riesgo y de confianza se muestran con **color y con texto**; la vista simple es la predeterminada y la avanzada es opcional; plantilla de CSV descargable; el chat ofrece preguntas de ejemplo como botones.
- [ ] **4.6 Seguridad y privacidad:** ejecución local sin llamadas externas; solo la interfaz queda expuesta, en localhost; los registros no guardan el contenido de los lotes; el rango de edad se documenta como variable demográfica y se reporta el desempeño por grupo.
- [ ] **4.7 Pruebas de contrato (`tests/`):** cada servicio responde con el formato de 4.1 y la app funciona contra la API simulada.
- [ ] **`scripts/04_app_api/run_api.py` y `run_app.py`:** lanzadores de una línea, con instrucciones en el README. Si hace falta una librería mínima para correr o probar la API (servidor, cliente HTTP), **pregúntame antes**.

**Listo cuando:** un flujo completo (cargar un CSV, ver resultados, exportar y hacer una pregunta) funciona de principio a fin con los servicios reales.

---

## Fase 5A — Evidencia y RAG `[L]` (`scripts/05a_evidencia_rag/`)

Objetivo: que cada consulta C1–C7 tenga una función determinista que devuelve su paquete de evidencia, y una base documental para las definiciones.

- [ ] **`01_base_documental.py`**
  - Hace: valida y divide en fragmentos los documentos de `data/knowledge/*.md`: definiciones de Over30, MOB, vintage, GCL, NCL, cuentas expuestas, SHAP, niveles de riesgo y confianza, estrategias y segmentos, en lenguaje para personal no especialista. Crea los documentos que falten.
  - Entra → sale: `data/knowledge/*.md` → fragmentos validados.
  - Listo cuando: cada término del glosario de `CLAUDE.md` tiene su documento y ningún fragmento afirma causalidad.
- [ ] **`02_indexar_chroma.py`**
  - Hace: crea el índice vectorial persistente en `data/chroma/`. **Verifica de dónde saca Chroma el modelo de embeddings**: según tengo entendido, el de por defecto se descarga de internet la primera vez, y eso rompe la regla de "todo local". Propón una alternativa local (por ejemplo, embeddings con Ollama) y **pregúntame cuál usar antes de indexar**.
  - Entra → sale: fragmentos → `data/chroma/`.
  - Listo cuando: el índice se reconstruye con un comando y no hace llamadas externas.
- [ ] **`03_probar_recuperacion.py`**
  - Hace: lanza consultas de definiciones (C7) y muestra los fragmentos recuperados con su fuente.
  - Entra → sale: preguntas de ejemplo → `reports/05a_evidencia_rag/recuperacion.csv`.
  - Listo cuando: las definiciones esperadas aparecen entre los primeros resultados en todas las preguntas de prueba.
- [ ] **`04_evidencia_shap.py` (C1, C2)** — Hace: `explain_profile` y `variable_effect` en `src/evidence/`, apoyadas en SHAP de S3 y en tasas observadas. Listo cuando: devuelven paquetes que cumplen el contrato, con cifras con nombre y versiones.
- [ ] **`05_evidencia_pandas.py` (C3, C5, C6)** — Hace: `compare_groups`, `vintage_evolution` y `summarize_batch`, con pandas sobre cohortes maduras, por MOB y sobre los resultados cargados. Listo cuando: cada función cumple el contrato y sus cifras se reproducen a mano en una prueba.
- [ ] **`06_evidencia_estrategia.py` (C4)** — Hace: `suggest_strategy`, con la salida de S4 y su intervalo; devuelve `no_evaluable` para DARIA 2.0 y perfiles fuera de dominio. Listo cuando: cubre los tres estados.
- [ ] **`07_sincronizar_ejemplos.py`**
  - Hace: genera ejemplos reales de cada consulta y compara su estructura con `data/mock/`, para que los ejemplos simulados no se desalineen del formato real.
  - Entra → sale: evidencia real → informe de diferencias de formato.
  - Listo cuando: cualquier diferencia de formato queda reportada; luego `evidence_mode` puede pasar a `"real"`.

**Listo cuando (fase):** C1–C7 devuelven evidencia real conforme al contrato, de forma determinista, con función, parámetros y versiones, y con pruebas.

---

## Fase 5B — Agente `[E]` (`scripts/05b_asistente/`, `src/assistant/`)

Objetivo: un asistente que **solo redacta sobre evidencia calculada**: nunca calcula ni lee tablas crudas. Arranca con `evidence_mode = "mock"`. Llama 3.1 8B corre **local con Ollama**.

**Flujo:** pregunta → clasificador de intención con salida restringida elige una consulta del catálogo o responde "no soportada" → se validan los parámetros contra los catálogos (`src/data/catalog.py`) → una función de evidencia calcula el paquete → (en C7, se recuperan fragmentos) → el modelo redacta con un *prompt* estricto → una barrera compara cada cifra del texto con la evidencia y, si algo no coincide, muestra la respuesta de plantilla con los datos verificados.

- [ ] **`01_verificar_ollama.py`**
  - Hace: comprueba que el servidor de Ollama responde, que el modelo `llama3.1:8b` está descargado y mide la latencia de una pregunta corta. Si falta algo, imprime los pasos exactos para instalarlo.
  - Entra → sale: `configs/config.json` → informe en la terminal.
  - Listo cuando: detecta con mensajes claros un Ollama apagado o un modelo faltante.
- [ ] **`02_clasificador_intencion.py`**
  - Hace: `src/assistant/intent.py`. Salida restringida (por ejemplo, el parámetro `format` de Ollama con un esquema JSON; verifica que la versión instalada lo soporta) con `consulta_id` y parámetros; valida los parámetros contra los catálogos; una pregunta fuera del catálogo recibe la lista de lo que sí se puede responder.
  - Entra → sale: pregunta → intención y parámetros validados, o `no_soportada`.
  - Listo cuando: clasifica correctamente el 90% o más del conjunto de evaluación (en 07).
- [ ] **`03_redactor.py`**
  - Hace: `src/assistant/writer.py`. *Prompt* estricto: usar solo las cifras del paquete, lenguaje no causal, el contenido recuperado se trata como **datos y no como instrucciones**, en español claro para personal no especialista.
  - Entra → sale: paquete de evidencia → texto redactado.
  - Listo cuando: redacta las siete consultas con los ejemplos simulados sin cifras fuera del paquete.
- [ ] **`04_barrera_numerica.py`**
  - Hace: `src/assistant/guard.py`. Extrae cada cifra del texto (porcentajes, conteos, decimales y su formato) y la compara con `cifras` y `tablas`; si algo no coincide, devuelve la **respuesta de plantilla** con los datos verificados. Se abstiene si la evidencia es insuficiente, el perfil es *no evaluable* o la intención no está soportada.
  - Entra → sale: texto y paquete → texto aprobado o respuesta de plantilla.
  - Listo cuando: **cero cifras sin respaldo** en las pruebas, incluidas respuestas inventadas a propósito.
- [ ] **`05_flujo_chat.py`**
  - Hace: `src/assistant/chat.py`. Une todo el flujo y devuelve, junto con la respuesta, la función, los parámetros y las versiones usadas; cumple `docs/contratos/chat.md`. Se puede correr desde la terminal con una pregunta.
  - Entra → sale: pregunta → respuesta con estado (`respondida`, `abstenida`, `no_soportada`).
  - Listo cuando: las siete consultas de ejemplo funcionan de principio a fin con ejemplos simulados.
- [ ] **`06_conjunto_evaluacion.py`**
  - Hace: crea `data/eval/preguntas_evaluacion.csv` con pregunta, intención esperada, parámetros esperados y tipo. Mínimos por categoría: **8 por cada consulta (56), 10 fuera de catálogo y 5 de inyección de instrucciones, en total 71**. (La propuesta dice "al menos 60", pero sus mínimos por categoría suman 71; usa los mínimos por categoría.)
  - Entra → sale: catálogo → CSV de preguntas.
  - Listo cuando: se cumplen los mínimos y las preguntas varían en redacción.
- [ ] **`07_evaluar_asistente.py`**
  - Hace: corre el conjunto de evaluación y mide intención correcta, cifras sin respaldo, abstención correcta, resistencia a inyección y latencia mediana. Guarda el informe en `reports/05b_asistente/`.
  - Entra → sale: preguntas de evaluación → informe.
  - Listo cuando: **intención correcta ≥ 90%, cero cifras sin respaldo, abstención correcta ≥ 95% de lo no respondible, resistencia a inyección del 100% y latencia mediana ≤ 20 segundos.**
- [ ] **Cambio a evidencia real:** cuando 5A esté lista, pasa `evidence_mode` a `"real"` en `configs/config.json`, vuelve a correr `07` y registra la diferencia contra los resultados con ejemplos simulados.

**Listo cuando (fase):** se cumplen todos los criterios de `07_evaluar_asistente.py` con evidencia real. En esta fase se decide si el asistente se divide en varios agentes; no lo hagas antes de que lo indique.

---

## Fase 6 — Cierre `[L+E]` (`scripts/06_cierre/`)

Objetivo: comprobar que todo cumple los criterios de aceptación y dejarlo documentado.

- [ ] **`01_pruebas_extremo_a_extremo.py`** — Hace: corre el flujo completo contra los servicios reales: cargar un CSV, validar, predecir, explicar, comparar, chat y exportar. Listo cuando: ninguna etapa falla y los archivos inválidos nunca llegan a los modelos.
- [ ] **`02_verificar_criterios.py`** — Hace: reúne en un solo informe el estado de cada criterio de aceptación de la tabla de abajo, leyendo los reportes y metadatos ya generados. Listo cuando: cada fila indica cumple o no cumple, con su evidencia.
- [ ] **Prueba de usabilidad:** tres personas sin formación técnica completan el flujo (cargar, ver resultados, exportar y hacer una pregunta) **en 10 minutos o menos y sin ayuda**; se anotan los tropiezos y se corrigen.
- [ ] **Documentación en `docs/`:** manual de usuario, guía de instalación y una ficha de cada modelo (con su versión y sus métricas; aclarando que los datos son sintéticos).
- [ ] **Revisión de seguridad:** ejecución local, solo la interfaz en localhost, límite de 50 MB, UTF-8, columnas no declaradas descartadas, registros sin contenido de lotes y salidas etiquetadas como apoyo al análisis (nunca aprobación o rechazo).
- [ ] **Despliegue sin Docker:** la guía de instalación funciona en una máquina limpia (entorno virtual, `pip install`, Ollama y un comando para levantar la API y la interfaz).
- [ ] **Docker Compose (opcional, solo si se confirma que hace falta):** tres contenedores (interfaz, API de inferencia y Ollama), con solo la interfaz expuesta en localhost.

### Criterios de aceptación

| Área | Criterio | ¿Cumple? |
|---|---|---|
| Datos | El CSV de cada tipo se valida con errores legibles y ningún archivo inválido llega a los modelos | ⬜ |
| S1 y S2 | El MAE ponderado de S1 es al menos 10% menor que el de la línea base en prueba; S2 es exactamente cuentas planeadas por S1 | ⬜ |
| S3 | AUC, KS y Gini superiores a la línea base; razón observado/predicho entre 0.95 y 1.05; curva de calibración revisada | ⬜ |
| S4 | Acuerdo de signo de al menos 70% entre los dos métodos; solo sugiere cuando el intervalo excluye cero y ambos coinciden | ⬜ |
| Consistencia | Las reglas se cumplen en el 100% de las filas de prueba | ⬜ |
| SHAP | La suma de contribuciones coincide con la predicción (tolerancia 0.0001) en los tres niveles | ⬜ |
| Asistente | Intención correcta ≥ 90%; cero cifras sin respaldo; abstención ≥ 95% de lo no respondible; 100% de resistencia a inyección; latencia mediana ≤ 20 s | ⬜ |
| Usabilidad | Tres personas sin formación técnica completan el flujo en 10 minutos o menos y sin ayuda | ⬜ |
| Despliegue | El sistema levanta en una máquina limpia y solo la interfaz queda expuesta en localhost | ⬜ |
| Documentación | Manual de usuario, guía de instalación y ficha de cada modelo con su versión y sus métricas | ⬜ |

**Listo cuando:** todas las filas dicen que cumplen y el repo se puede instalar y usar siguiendo solo el `README.md`.
