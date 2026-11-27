# Contrato: servicio de chat

> **Estado:** borrador de la Fase 0b, para revisar entre Luis y Ernesto. Cambiarlo requiere acuerdo de ambos.
> **Lado que implementa:** Ernesto (`src/assistant/chat.py`, Fase 5B). **Lados que lo usan:** la API (`api/`) y la app (`app/`), Fase 4.
> **Depende de:** [evidencia.md](evidencia.md) (formato del paquete de evidencia).

El chat recibe una pregunta en español, la clasifica en una consulta del catálogo (C1 a C7), pide la evidencia, redacta y pasa la barrera numérica. **Nunca** calcula cifras por su cuenta.

---

## 1. Entrada

```json
{
  "pregunta": "¿Por qué este perfil tiene riesgo alto?",
  "perfil": {
    "estrategia": "r.Reca champ",
    "segmento_estrategia": "Costco",
    "producto": "Costco",
    "canal": "Costco",
    "rango_edad": "02| Young 25-30",
    "rango_score": "141-150"
  },
  "lote_id": null
}
```

| Campo | Tipo | Obligatorio | Qué contiene |
|---|---|---|---|
| `pregunta` | texto | Sí | La pregunta del usuario, en lenguaje natural. |
| `perfil` | objeto o `null` | No | El perfil en sesión, con las claves de `perfil` en [evidencia.md](evidencia.md#parámetros). Lo usan C1 y C4 cuando la pregunta no trae el perfil. |
| `lote_id` | texto o `null` | No | El lote cargado en sesión. Lo usa C6. |

---

## 2. Salida

```json
{
  "respuesta": "La tasa de impago estimada a 6 meses para este perfil es 8.7%, nivel de riesgo Alto. ...",
  "estado": "respondida",
  "consulta_id": "C1",
  "funcion": "evidence.explain_profile",
  "parametros": {"perfil": {"estrategia": "r.Reca champ", "...": "..."}},
  "versiones": {"modelo_s3": "s3/simulado", "datos_sha256": "simulado", "horizonte_mob": 6},
  "estado_evidencia": "ok",
  "motivo": null,
  "barrera": {"aprobada": true, "cifras_sin_respaldo": [], "uso_plantilla": false},
  "consultas_disponibles": [],
  "simulado": true
}
```

| Campo | Tipo | Qué contiene |
|---|---|---|
| `respuesta` | texto | El texto que ve el usuario, en español claro. Si la barrera rechaza el texto redactado, es la respuesta de plantilla con los datos verificados. |
| `estado` | texto | `respondida`, `abstenida` o `no_soportada` (ver §3). |
| `consulta_id` | texto o `null` | `C1` a `C7`; `null` si la pregunta no está soportada. |
| `funcion` | texto o `null` | La función de evidencia usada (`evidence.<nombre>`). |
| `parametros` | objeto | Los parámetros con los que se pidió la evidencia (ya validados contra los catálogos). |
| `versiones` | objeto | Copia de `versiones` del paquete de evidencia. La app las muestra en cada respuesta. |
| `estado_evidencia` | texto o `null` | Copia de `estado` del paquete (`ok`, `no_evaluable`, `sin_evidencia`); `null` si no se pidió evidencia. |
| `motivo` | texto o `null` | Por qué el chat se abstuvo o no pudo responder. |
| `barrera` | objeto | `aprobada` (si el texto redactado pasó), `cifras_sin_respaldo` (las cifras rechazadas, como texto) y `uso_plantilla` (si se mostró la respuesta de plantilla). |
| `consultas_disponibles` | lista | Solo con `no_soportada`: ejemplos de lo que sí se puede preguntar (uno por consulta del catálogo). |
| `simulado` | booleano | Copia de `simulado` del paquete; la app avisa cuando la respuesta usa datos simulados. |

---

## 3. Estados

| Estado | Cuándo | Qué muestra |
|---|---|---|
| `respondida` | La pregunta corresponde a una consulta del catálogo y la evidencia tiene `estado = "ok"`. | El texto redactado si pasó la barrera; si no, la respuesta de plantilla (`barrera.uso_plantilla = true`). |
| `abstenida` | La evidencia tiene `estado` `no_evaluable` o `sin_evidencia`, o los parámetros no se pudieron validar. | El `motivo` en lenguaje claro, sin cifras. |
| `no_soportada` | El clasificador no encuentra una consulta del catálogo (incluidos los intentos de inyección de instrucciones). | Un mensaje breve y la lista `consultas_disponibles`. |

---

## 4. Ejemplos

**Abstenida (DARIA 2.0 no es evaluable en riesgo):**

```json
{
  "respuesta": "No puedo estimar el riesgo de este perfil: DARIA 2.0 solo tiene MOB 0 y 1, así que todavía no hay desempeño de riesgo a 6 meses.",
  "estado": "abstenida",
  "consulta_id": "C1",
  "funcion": "evidence.explain_profile",
  "parametros": {"perfil": {"estrategia": "DARIA 2.0", "...": "..."}},
  "versiones": {"modelo_s1": "s1/simulado", "modelo_s3": "s3/simulado", "datos_sha256": "simulado", "horizonte_mob": 6},
  "estado_evidencia": "no_evaluable",
  "motivo": "DARIA 2.0 solo tiene MOB 0 y 1: tiene línea asignada pero todavía no hay desempeño de riesgo a 6 meses.",
  "barrera": {"aprobada": true, "cifras_sin_respaldo": [], "uso_plantilla": false},
  "consultas_disponibles": [],
  "simulado": true
}
```

**No soportada:**

```json
{
  "respuesta": "Esa pregunta está fuera de lo que puedo responder. Puedes preguntarme, por ejemplo:",
  "estado": "no_soportada",
  "consulta_id": null,
  "funcion": null,
  "parametros": {},
  "versiones": {},
  "estado_evidencia": null,
  "motivo": "La pregunta no corresponde a ninguna consulta del catálogo.",
  "barrera": {"aprobada": true, "cifras_sin_respaldo": [], "uso_plantilla": false},
  "consultas_disponibles": [
    "¿Por qué este perfil tiene riesgo alto?",
    "¿Cómo afecta el canal a la tasa de impago?",
    "Compara Costco con Home Depot en Over30",
    "¿Qué estrategia conviene para este perfil?",
    "¿Cómo evoluciona Over30 en la vintage 202506?",
    "Resume los resultados del lote",
    "¿Qué significa GCL?"
  ],
  "simulado": false
}
```

---

## 5. Pendiente de acordar

- Si la respuesta incluye las `tablas` del paquete para que la app las muestre, o solo el texto.
- El tiempo máximo de espera de Ollama (`ollama.timeout_seconds` en `configs/config.json`) y qué responde el chat si se agota.
- Si se guarda un historial de la conversación (los registros no deben guardar el contenido de los lotes).
