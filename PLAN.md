# PLAN.md — Plan de trabajo

> **Cómo se usa:** cada tarea es un script (o una pieza de `src/`). Al terminar una, se marca su casilla `[x]` y se anota debajo la decisión o el resultado clave, si lo hay. No se pasa a la siguiente fase sin el visto bueno de su responsable.
> **Responsables:** `[L]` Luis · `[E]` Ernesto · `[L+E]` ambos. El avance se mide por tareas terminadas.
> **Otros documentos:** reglas y decisiones en [CLAUDE.md](CLAUDE.md); instalación, scripts y dónde está cada cosa en [README.md](README.md).

---

## 1. Vista general

| Fase | Qué se hace | Quién | Estado |
|---|---|---|---|
| 0. Preparación | Repo, entorno, configuración base | L | ✅ |
| 0b. Reestructura | Scripts en vez de notebooks, README, bloqueo de git, contratos y ejemplos simulados | L | ✅ |
| 1. Datos | Cargar, validar y entender la base | L | ✅ |
| 2. Modelos S1, S2 y S3 | Líneas base y XGBoost, métricas y versionado | L | ✅ |
| 3. Estrategia, consistencia y explicabilidad | S4, reglas de consistencia, SHAP y factores de S1 | L | ✅ |
| 4. App y API | FastAPI + Streamlit | L+E | ⬜ |
| 5A. Evidencia y RAG | Funciones deterministas C1–C7 y base documental | L | ⬜ |
| 5B. Agente | Intención, redacción, barrera numérica, chat y evaluación | E | En curso, con ejemplos simulados |
| 6. Cierre | Pruebas de extremo a extremo, documentación y criterios | L+E | ⬜ |

**Qué depende de qué**

```
0b ──► 1 ──► 2 ──► 3 ──► 4 ──► 6
 │                 └──► 5A ──┘   ▲
 └──────────► 5B (con mocks) ────┘
```

- Ernesto **no espera** a Luis: trabaja con los ejemplos de `data/mock/`. Cuando 5A entregue la evidencia real, `evidence_mode` pasa de `"mock"` a `"real"` en `configs/config.json`.
- Los dos frentes se comunican por los contratos de `docs/contratos/`: [evidencia.md](docs/contratos/evidencia.md) (catálogo C1–C7 y formato del paquete) y [chat.md](docs/contratos/chat.md). Cada quien decide dentro de su frente; lo que choque al unir los trabajos se resuelve en ese momento.

---

## 2. Cierre de cada script (aplica a todos)

- El docstring inicial dice objetivo, entradas, salidas, cómo correrlo y "listo cuando".
- Cada bloque de código tiene su comentario (primero qué hace, luego por qué), y las fórmulas están traducidas a código.
- Las gráficas están en funciones aparte y se omiten con `--sin-graficas`.
- Corre completo desde la raíz, sin intervención, y repetirlo da el mismo resultado.
- La lógica reutilizable está en `src/` y tiene prueba en `tests/`.
- `README.md` (§6, Scripts) se actualiza si se agregó un script o una carpeta.
- Se marca la casilla en este archivo y se hace el commit.

---

## 3. Fases terminadas

### Fase 0 — Preparación ✅

- [x] Estructura de carpetas, `.gitignore`, entorno virtual y `requirements/` (`base.txt`, `ml.txt`, `lock.txt`).
- [x] `configs/config.json` (semilla 42, H = 6, particiones, rutas) y `src/utils/` (`load_config`, `get_path`, `set_seed`), con pruebas.
- [x] Datos en `data/raw/`, diccionario en `data/contracts/`, propuesta en `docs/` y primer commit.

### Fase 0b — Reestructura `[L]` ✅

- [x] `notebooks/` reemplazado por `scripts/` (subcarpetas `01_datos` … `06_cierre`); no queda ningún `.ipynb`. Las dependencias de Jupyter se quitaron de `requirements/`.
- [x] Carpetas nuevas: `reports/`, `data/mock/`, `knowledge/`, `eval/`, `scratch/`, `docs/contratos/`, `src/strategy/`, `evidence/`, `rag/`.
- [x] `.gitignore`: datos, reportes, `scratch/`, Chroma y binarios de `models/` fuera de git; sí se versionan `metadata.json` y `actual.json`.
- [x] `scripts/_plantilla.py` y `scripts/run_fase.py` (corre una fase en orden, `--desde NN`, se detiene en el primer error).
- [x] `configs/config.json` con `evidence_mode = "mock"`, Ollama y Chroma.
- [x] Bloqueo técnico de los comandos de git prohibidos en `.claude/settings.json`.
- [x] Contratos `docs/contratos/evidencia.md` y `chat.md`; ejemplos simulados `data/mock/evidencia_*.json` (C1–C7, `no_evaluable` y `sin_evidencia`, con `"simulado": true`).
- [x] `src/evidence/get_evidence()`: sirve los ejemplos en modo `mock`; en modo `real` lanza `NotImplementedError`. Pruebas de contrato en `tests/test_evidence_contract.py`.
- [x] `README.md` completo, sin fechas ni menciones a notebooks.

### Fase 1 — Datos `[L]` ✅ (`scripts/01_datos/`)

Hallazgos y decisiones en [docs/datos_fase1.md](docs/datos_fase1.md).

- [x] **`01_carga_y_diccionario.py`** — carga el CSV con nombres reales o alias `dummiN`, aplica `.str.strip()` a las categorías y construye y verifica `src/data/catalog.py`.
  - *Resultado:* el CSV trae los nombres reales; `MOB` se llama `MOB`.
- [x] **`02_validacion_contrato.py`** — `src/data/validation.py` con pandas puro, con las reglas bloqueantes y los avisos de CLAUDE.md §5.
  - *Resultado:* la base pasa todas las reglas.
  - *Decisión:* un valor nulo es falla bloqueante.
  - *Columnas acumuladas:* `OVER30ACCTS`, `OVER30AMTS`, `GCLAMTS`, `NCLAMTS`, `RECOVERYAMTS` y `REVENUE`.
- [x] **`03_exploracion_cohortes.py`** — cuentas por vintage, estrategia y segmento; producto, canal, score, edad y filas por MOB.
- [x] **`04_madurez_y_curvas.py`** — madurez a H = 6 (`src/data/cohorts.py`), curvas de Over30 y los datasets `s1_dataset.csv` y `s3_dataset.csv`.
  - *Resultado:* 27 vintages maduras.
  - *Decisión:* `OVER30ACCTS > ENRACCTS` ocurre en 0 de 52,835 cohortes, así que el denominador de S3 sigue siendo `ENRACCTS`.
- [x] **`05_balance_entre_estrategias.py`** — balance r.Reca champ vs. DARIA 1.6 en 202511–202607 y marca de DARIA 2.0 como no evaluable en riesgo.
  - *Decisión:* S4 compara solo en el soporte común. DARIA 1.6 no tiene Costco ni Beyond.

### Fase 2 — Modelos S1, S2 y S3 `[L]` ✅ (`scripts/02_modelos_prediccion/`)

Métricas y criterios en [docs/fichas/ficha_s1.md](docs/fichas/ficha_s1.md) y [ficha_s3.md](docs/fichas/ficha_s3.md).

**Modelos de reserva (decisión de Luis):** los GLM de S1 y S3 se guardan como reserva de emergencia. Los principales siguen siendo los publicados.
- Se guardan en JSON con `src/models/glm.py`: solo el intercepto y los coeficientes.
- Para activar uno, se apunta `models/<salida>/actual.json` a su versión, o se carga con `load_s1(models_dir, versión)` / `load_s3(...)`.
- La explicabilidad de la Fase 3 está hecha para los modelos publicados y no aplica a las reservas.

- [x] **`00_variables_y_particion.py`** — variables de entrada (`src/features/engineering.py`), partición temporal por vintage y prueba automática de fuga.
- [x] **`01_s1_linea_base.py`** — media ponderada por producto × score y GLM Gamma.
  - *Reserva:* el GLM Gamma queda guardado como modelo de reserva de S1 (`s1-767020790c`), sin publicar.
- [x] **`02_s1_xgboost.py`** — XGBoost sobre el log de la línea, con Optuna.
  - *Resultado:* MAE de prueba 2.9% **peor** que la base simple; no cumple el criterio (≥ 10% mejor).
  - *Decisión de Luis (opción A):* se publica la **línea base simple** como S1. El XGBoost queda guardado como registro (`s1-4eeb609786`).
- [x] **`03_s2_monto_total.py`** — S2 = cuentas planeadas × S1 (`src/models/s2.py`), exacto.
- [x] **`04_s3_linea_base.py`** — tasa por score × producto y GLM binomial, con métricas ponderadas.
  - *Reserva:* el GLM binomial queda guardado como modelo de reserva de S3 (`s3-f8f7c31415`), sin publicar y con sus propios umbrales de riesgo.
- [x] **`05_s3_xgboost.py`** — XGBoost con etiqueta fraccional, monótono decreciente en el score y calibración Platt en validación.
  - *Resultado:* aprobado y publicado (`s3-e212982330`).
  - *Umbrales de riesgo:* percentiles de la tasa predicha en entrenamiento, guardados en su `metadata.json`.
- [x] **`06_resumen_modelos.py`** — tabla resumen y fichas generadas desde `metadata.json`.

### Fase 3 — Estrategia, consistencia y explicabilidad `[L]` ✅ (`scripts/03_estrategia_y_explicabilidad/`)

Decisiones de S4 y consistencia en [docs/consistencia_fase3.md](docs/consistencia_fase3.md).

**3A. Estrategia (S4)**

- [x] **`01_ventana_aleatoria.py`** — ventana aleatoria 202511–202607, cohortes maduras, perfiles producto × canal × edad y soporte común. Fuera del soporte: "Sin evidencia". DARIA 2.0 queda excluida.
- [x] **`02_s4_observado_bootstrap.py`** — método 1: diferencia observada DARIA 1.6 − r.Reca champ, con intervalo bootstrap sobre cohortes (semilla 42).
- [x] **`03_s4_prediccion_escenarios.py`** — método 2: se predicen con S3 las cohortes reales de cada estrategia y se promedian ponderando por `ENRACCTS`.
  - *Decisión:* cada vintage de la ventana se predice con un S3 reentrenado **sin ella**, con los mismos hiperparámetros y la misma calibración, para no inflar la concordancia.
- [x] **`04_s4_sugerencia_y_validacion.py`** — sugiere solo si el intervalo del método 1 excluye cero y los dos métodos coinciden en el signo; si no, "Sin diferencia concluyente".
  - *Resultado:* concordancia 92/93 = 98.9% (criterio ≥ 70%).
  - *Sugerencias:* r.Reca champ en 91 perfiles, DARIA 1.6 en 1, sin diferencia concluyente en 102 y sin evidencia en 28.

**3B. Consistencia**

- [x] **`05_reglas_consistencia.py`** — `src/consistency/`, que corre después de los modelos:
  - calcula S2;
  - recorta las tasas a [0, 1];
  - deja sin salida un perfil fuera de dominio;
  - asigna el nivel de riesgo, la alerta de Crítico y la confianza;
  - agrega las versiones.
  - *Decisiones:*
    - **Fuera de dominio:** la combinación (familia, segmento, producto, canal) no aparece en el entrenamiento del modelo.
    - **Perfiles similares:** misma familia, segmento, producto, canal y edad, con cualquier score.
    - **Confianza de DARIA 2.0:** "No aplica", porque no tiene tasa.
    - **Umbrales de riesgo:** se conservan los de la Fase 2. Es una desviación documentada de la propuesta, que pedía la tasa histórica observada.
  - *Resultado:* las 12 reglas se cumplen en el 100% de las 19,235 filas de prueba.

**3C. Explicabilidad**

- [x] **`06_shap_global.py`** — SHAP de S3 con el TreeSHAP nativo de XGBoost (`pred_contribs`), sumado por variable original; factores exactos de S1; barras de importancia y *beeswarm*.
- [x] **`07_shap_segmento.py`** — barras y *beeswarm* de S3 para cualquier segmento (`--variable`, `--valor`).
- [x] **`08_shap_perfil.py`** — cascada de un perfil (S1 por factores, S3 por SHAP) y su tabla de contribuciones.
- [x] **`09_frases_lectura.py`** — los tres factores que más suben y los tres que más bajan, por plantilla. Todas llevan la nota *"Estas explicaciones describen cómo usa las variables el modelo; no establecen relaciones causales."*
  - *Resultado:* ninguna frase usa lenguaje causal y las 8 gráficas llevan la nota.
- [x] **`10_verificar_aditividad.py`** — base + Σ contribuciones = salida, con tolerancia 0.0001, en los niveles global, segmento y perfil. En S3 se compara con la salida **sin calibrar**; en S1, con el log de la línea.

> **Por qué así:**
> - **TreeSHAP nativo en vez de `shap.TreeExplainer`:** shap 0.49.1 no lee el `base_score` de XGBoost 3.x. Es el mismo algoritmo, y la aditividad lo verifica.
> - **Cómo se explica S1:** como el S1 publicado es una tabla producto × score, su explicación es exacta. Línea = promedio global × factor del producto × factor del score dentro del producto (`src/explain/s1_factors.py`).

- [x] Gráficas de S1 revisadas por Luis.
- [x] **Revisión general de Luis** de S4, consistencia y explicabilidad: aprobada; fase cerrada.

---

## 4. Fases por hacer

### Fase 4 — App y API `[L+E]` (`api/`, `app/`, `scripts/04_app_api/`)

Objetivo: una interfaz usable por personal sin formación técnica, con servicios locales. **Antes de empezar, decidamos entre Luis y Ernesto quién hace qué** (propuesta: Luis, servicios de validar, predecir, explicar y comparar; Ernesto, el servicio de chat y su pantalla).

- [ ] **4.0 `requirements/app.txt`** (streamlit, fastapi, pydantic) y sus versiones en `lock.txt`, comprobadas en Python 3.10.
- [ ] **4.1 Contratos de la API (`api/schemas.py`, Pydantic):** entrada y salida de cada servicio. **Entrada de perfiles:** obligatorias estrategia, segmento de estrategia, producto, canal, rango de edad y rango de score; opcionales un identificador de perfil, la vintage (solo para derivar el mes de booking) y las cuentas planeadas (habilitan el monto total). **Entrada histórica:** el esquema del diccionario, para comparar lo observado contra lo predicho. **Salida:** atributos del perfil, línea promedio, monto total, tasa de impago esperada, nivel de riesgo, estrategia sugerida con su intervalo, confianza, estado con su motivo, y versiones de modelo y de datos con el horizonte.
- [ ] **4.2 API simulada (`api/main.py`):** devuelve el formato final con datos de ejemplo, para que la app y el agente no esperen a los modelos.
- [ ] **4.3 Servicios reales (FastAPI, solo localhost):** validar, predecir, explicar, comparar estrategias, chat y estado del sistema. Validación con las reglas de `src/data/validation.py`; los errores indican qué falló, en qué fila y qué hacer. Archivos admitidos de hasta **50 MB en UTF-8**; las columnas no declaradas se descartan. **Nota:** la base histórica completa pesa unos 193 MB, así que decide con Luis cómo se maneja el modo histórico (por ejemplo, muestras o agregados).
- [ ] **4.4 Interfaz (Streamlit):** carga de CSV, tablero, gráficas (Plotly), comparador de estrategias, chat y exportación a CSV con las columnas de salida de 4.1.
- [ ] **4.5 Usabilidad:** cada término técnico (Over30, MOB, vintage, SHAP) tiene una definición breve al pasar el cursor; cada resultado trae una frase de lectura en español generada por plantilla; los niveles de riesgo y de confianza se muestran con **color y con texto**; la vista simple es la predeterminada y la avanzada es opcional; plantilla de CSV descargable; el chat ofrece preguntas de ejemplo como botones.
- [ ] **4.6 Seguridad y privacidad:** ejecución local sin llamadas externas; solo la interfaz queda expuesta, en localhost; los registros no guardan el contenido de los lotes; el rango de edad se documenta como variable demográfica y se reporta el desempeño por grupo.
- [ ] **4.7 Pruebas de contrato (`tests/`):** cada servicio responde con el formato de 4.1 y la app funciona contra la API simulada.
- [ ] **`scripts/04_app_api/run_api.py` y `run_app.py`:** lanzadores de una línea, con instrucciones en el README. Si hace falta una librería mínima para correr o probar la API (servidor, cliente HTTP), **pregúntame antes**.

**Listo cuando:** un flujo completo (cargar un CSV, ver resultados, exportar y hacer una pregunta) funciona de principio a fin con los servicios reales.

### Fase 5A — Evidencia y RAG `[L]` (`scripts/05a_evidencia_rag/`)

Objetivo: que cada consulta C1–C7 tenga una función determinista que devuelve su paquete de evidencia, y una base documental para las definiciones.

- [ ] **`requirements/assistant.txt`** (chromadb, ollama) y sus versiones en `lock.txt`, comprobadas en Python 3.10. Lo comparten 5A y 5B: lo crea quien llegue primero.
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

### Fase 5B — Agente `[E]` (`scripts/05b_asistente/`, `src/assistant/`)

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

### Fase 6 — Cierre `[L+E]` (`scripts/06_cierre/`)

Objetivo: comprobar que todo cumple los criterios de aceptación y dejarlo documentado.

- [ ] **`01_pruebas_extremo_a_extremo.py`** — Hace: corre el flujo completo contra los servicios reales: cargar un CSV, validar, predecir, explicar, comparar, chat y exportar. Listo cuando: ninguna etapa falla y los archivos inválidos nunca llegan a los modelos.
- [ ] **`02_verificar_criterios.py`** — Hace: reúne en un solo informe el estado de cada criterio de aceptación de la tabla de abajo, leyendo los reportes y metadatos ya generados. Listo cuando: cada fila indica cumple o no cumple, con su evidencia.
- [ ] **Prueba de usabilidad:** tres personas sin formación técnica completan el flujo (cargar, ver resultados, exportar y hacer una pregunta) **en 10 minutos o menos y sin ayuda**; se anotan los tropiezos y se corrigen.
- [ ] **Documentación en `docs/`:** manual de usuario, guía de instalación y una ficha de cada modelo (con su versión y sus métricas; aclarando que los datos son sintéticos).
- [ ] **Revisión de seguridad:** ejecución local, solo la interfaz en localhost, límite de 50 MB, UTF-8, columnas no declaradas descartadas, registros sin contenido de lotes y salidas etiquetadas como apoyo al análisis (nunca aprobación o rechazo).
- [ ] **Despliegue sin Docker:** la guía de instalación funciona en una máquina limpia (entorno virtual, `pip install`, Ollama y un comando para levantar la API y la interfaz).
- [ ] **Docker Compose (opcional, solo si se confirma que hace falta):** tres contenedores (interfaz, API de inferencia y Ollama), con solo la interfaz expuesta en localhost.


---

## 5. Criterios de aceptación

| Área | Criterio | Estado |
|---|---|---|
| Datos | El CSV se valida con errores legibles y ningún archivo inválido llega a los modelos | ✅ validador con pruebas (Fase 1); falta probarlo desde la app (Fase 6) |
| S1 y S2 | El MAE ponderado de S1 es al menos 10% menor que el de la línea base en prueba; S2 es exactamente cuentas planeadas × S1 | ❌ en S1: el XGBoost no lo cumplió y se publicó la línea base simple (decisión de Luis) · ✅ en S2 |
| S3 | No peor que el GLM (log loss y AUC); razón observado/predicho entre 0.95 y 1.05; tasa observada creciente por nivel de riesgo | ✅ AUC 0.6462 vs. 0.6454; obs/pred 1.012 |
| S4 | Acuerdo de signo de al menos 70% entre los dos métodos; solo sugiere cuando el intervalo excluye cero y ambos coinciden | ✅ 98.9% |
| Consistencia | Las reglas se cumplen en el 100% de las filas de prueba | ✅ 12 de 12 reglas |
| Explicabilidad | La suma de contribuciones coincide con la predicción (tolerancia 0.0001) en los tres niveles | ✅ S1 y S3 |
| Asistente | Intención correcta ≥ 90%; cero cifras sin respaldo; abstención ≥ 95% de lo no respondible; 100% de resistencia a inyección; latencia mediana ≤ 20 s | ⬜ |
| Usabilidad | Tres personas sin formación técnica completan el flujo en 10 minutos o menos y sin ayuda | ⬜ |
| Despliegue | El sistema levanta en una máquina limpia y solo la interfaz queda expuesta en localhost | ⬜ |
| Documentación | Manual de usuario, guía de instalación y ficha de cada modelo con su versión y sus métricas | Fichas ✅ · resto ⬜ |

**Listo cuando:** todas las filas cumplen (o tienen una decisión documentada) y el repo se puede instalar y usar siguiendo solo el `README.md`.
