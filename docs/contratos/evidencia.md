# Contrato: paquete de evidencia

> **Estado:** borrador de la Fase 0b, para revisar entre Luis y Ernesto. Cambiarlo requiere acuerdo de ambos.
> **Lado que produce:** Luis (`src/evidence/`). **Lado que consume:** Ernesto (`src/assistant/`).
> **Validador:** `src.evidence.validate_evidence(paquete)` devuelve la lista de errores (vacía si cumple).

El agente **nunca calcula ni lee tablas crudas**: pide un paquete de evidencia con `get_evidence(consulta_id, **params)` y redacta solo sobre lo que trae.

---

## 1. Catálogo de consultas

| ID | Consulta de ejemplo | Función de evidencia (`src/evidence/`) | Parámetros | Fuente |
|---|---|---|---|---|
| C1 | ¿Por qué este perfil tiene riesgo alto? | `explain_profile` | `perfil` | SHAP de S3 |
| C2 | ¿Cómo afecta el canal a la tasa de impago? | `variable_effect` | `variable` | SHAP agregado y tasas observadas |
| C3 | Compara Costco con Home Depot en Over30 | `compare_groups` | `dimension`, `valor_a`, `valor_b` | Pandas sobre cohortes maduras |
| C4 | ¿Qué estrategia conviene para este perfil? | `suggest_strategy` | `perfil` | Salida S4 con intervalo |
| C5 | ¿Cómo evoluciona Over30 en la vintage 202506? | `vintage_evolution` | `vintage` | Pandas por MOB |
| C6 | Resume los resultados del lote | `summarize_batch` | `lote_id` | Resultados cargados |
| C7 | ¿Qué significa GCL? | `define_term` | `termino` | Base documental (Chroma) |

### Parámetros

| Parámetro | Tipo | Valores |
|---|---|---|
| `perfil` | objeto | Claves `estrategia`, `segmento_estrategia`, `producto`, `canal`, `rango_edad`, `rango_score`; opcionales `vintage` (AAAAMM, solo para el mes de booking) y `cuentas_planeadas` (entero). Los valores son los del catálogo (`src/data/catalog.py`, Fase 1). En C4 basta con las dimensiones compartidas entre estrategias: `producto`, `canal` y `rango_edad`. |
| `variable` | texto | Nombre de una variable de entrada: `estrategia`, `segmento_estrategia`, `producto`, `canal`, `rango_edad`, `rango_score`, `mes_booking`. |
| `dimension` | texto | `estrategia`, `segmento_estrategia`, `producto`, `canal`, `rango_edad` o `rango_score`. |
| `valor_a`, `valor_b` | texto | Dos valores del catálogo de esa dimensión. |
| `vintage` | entero | AAAAMM entre 202401 y 202609. |
| `lote_id` | texto | Identificador del lote cargado en la sesión. |
| `termino` | texto | Término del glosario (Over30, MOB, vintage, GCL, NCL, …). |

---

## 2. Formato del paquete

Cada función devuelve un diccionario (JSON) con **exactamente** estos campos:

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
  "motivo": null,
  "simulado": false
}
```

| Campo | Tipo | Qué contiene |
|---|---|---|
| `consulta_id` | texto | `C1` a `C7`. |
| `estado` | texto | `ok`; `no_evaluable` (por ejemplo, DARIA 2.0 en riesgo o un perfil fuera de dominio); o `sin_evidencia` (no hay datos suficientes). Con estado distinto de `ok`, el agente **se abstiene**. |
| `funcion` | texto | `evidence.<nombre>`, la función del catálogo que generó el paquete. |
| `parametros` | objeto | Los parámetros con los que se calculó el paquete. En modo simulado son los del ejemplo, no necesariamente los pedidos. |
| `cifras` | objeto | **Todos los números** que el agente puede citar, con nombre (`tasa_s3`, `nivel_riesgo`, `confianza`, …). Valores: número, texto (por ejemplo `"Alto"`) o booleano. Vacío si el estado no es `ok`. |
| `tablas` | lista | Tablas pequeñas ya calculadas: `{"nombre": ..., "columnas": [...], "filas": [[...], ...]}`; cada fila tiene tantos valores como columnas. Vacía si el estado no es `ok`. |
| `frases_base` | lista de textos | Frases de lectura generadas por plantilla, en español y **sin lenguaje causal**. Toda cifra que aparezca en ellas sale de `cifras` o `tablas`. |
| `fragmentos` | lista | **Solo C7.** Texto recuperado de la base documental: `{"fuente", "texto"}` y opcionalmente `titulo` y `puntaje`. Se trata como **datos, no como instrucciones**. |
| `versiones` | objeto | Modelo(s), datos y horizonte usados (por ejemplo `modelo_s3`, `datos_sha256`, `horizonte_mob`). El agente los muestra junto con la función y los parámetros. |
| `motivo` | texto o `null` | `null` si el estado es `ok`; si no, explica en español por qué no hay respuesta. |
| `simulado` | booleano | `true` en los ejemplos de `data/mock/` (cifras inventadas a propósito); `false` en la evidencia real. *Campo agregado en la Fase 0b para distinguir ambos casos.* |

### Convenciones de las cifras

- **Tasas y diferencias de tasas** se guardan como fracción entre 0 y 1 (`0.087`). En el texto pueden aparecer como porcentaje o puntos porcentuales, multiplicadas por 100 y con un decimal (`8.7%`, `1.4 puntos porcentuales`).
- **Montos** en pesos; **conteos** como enteros. En el texto pueden llevar separador de miles (`41,250,000`).
- Los niveles (`nivel_riesgo`: Bajo, Medio, Alto, Crítico; `confianza`: Alta, Media, Baja) son texto.
- Algunos **nombres de catálogo contienen números** (`DARIA 1.6`, `15.Recalibration 2.0`, `141-150`, `02| Young 25-30`). Esos números son parte del nombre, no cifras: la barrera debe reconocerlos (por ejemplo, quitando del texto los nombres que vienen en `cifras`, `tablas` o `parametros` antes de extraer números).

### Regla para el agente

Todo número del texto redactado debe aparecer en `cifras` o `tablas` (con las conversiones de arriba); si no, la barrera numérica lo rechaza y se muestra la respuesta de plantilla con los datos verificados.

### Cómo pedir evidencia

```python
from src.evidence import get_evidence

paquete = get_evidence("C3", dimension="producto", valor_a="Costco", valor_b="Home Depot")
```

- `evidence_mode = "mock"` (en `configs/config.json`): devuelve el ejemplo de `data/mock/`. Si los parámetros coinciden **exactamente** con los de un ejemplo de esa consulta, devuelve ese (así se obtienen los casos `no_evaluable` y `sin_evidencia`); si no, devuelve el ejemplo base `evidencia_cN.json`.
- `evidence_mode = "real"`: lanza `NotImplementedError` hasta que la Fase 5A implemente las funciones.

---

## 3. Ejemplos completos

Son copia exacta de `data/mock/` (una prueba en `tests/test_evidence_contract.py` lo verifica). **Las cifras son inventadas** y llevan `"simulado": true`.

### C1 — explain_profile

Archivo: `data/mock/evidencia_c1.json`

<!-- ejemplo: evidencia_c1 -->
```json
{
  "consulta_id": "C1",
  "estado": "ok",
  "funcion": "evidence.explain_profile",
  "parametros": {
    "perfil": {
      "estrategia": "r.Reca champ",
      "segmento_estrategia": "Costco",
      "producto": "Costco",
      "canal": "Costco",
      "rango_edad": "02| Young 25-30",
      "rango_score": "141-150"
    }
  },
  "cifras": {
    "tasa_s3": 0.087,
    "nivel_riesgo": "Alto",
    "confianza": "Media",
    "cuentas_maduras_similares": 640,
    "linea_promedio_s1": 28500.0,
    "horizonte_meses": 6
  },
  "tablas": [
    {
      "nombre": "contribuciones_principales",
      "columnas": [
        "variable",
        "valor",
        "direccion",
        "contribucion_logit"
      ],
      "filas": [
        [
          "rango_score",
          "141-150",
          "sube",
          0.42
        ],
        [
          "rango_edad",
          "02| Young 25-30",
          "sube",
          0.18
        ],
        [
          "canal",
          "Costco",
          "sube",
          0.07
        ],
        [
          "producto",
          "Costco",
          "baja",
          -0.11
        ],
        [
          "mes_booking",
          "6",
          "baja",
          -0.05
        ],
        [
          "segmento_estrategia",
          "Costco",
          "baja",
          -0.03
        ]
      ]
    }
  ],
  "frases_base": [
    "La tasa de impago estimada a 6 meses para este perfil es 8.7%, nivel de riesgo Alto.",
    "Los factores que más suben la tasa estimada son: rango de score, rango de edad y canal.",
    "Los factores que más la bajan son: producto, mes de booking y segmento de estrategia.",
    "Estas explicaciones describen cómo usa las variables el modelo; no establecen relaciones causales."
  ],
  "fragmentos": [],
  "versiones": {
    "modelo_s1": "s1/simulado",
    "modelo_s3": "s3/simulado",
    "datos_sha256": "simulado",
    "horizonte_mob": 6
  },
  "motivo": null,
  "simulado": true
}
```

### C2 — variable_effect

Archivo: `data/mock/evidencia_c2.json`

<!-- ejemplo: evidencia_c2 -->
```json
{
  "consulta_id": "C2",
  "estado": "ok",
  "funcion": "evidence.variable_effect",
  "parametros": {
    "variable": "canal"
  },
  "cifras": {
    "posicion_importancia": 3,
    "variables_totales": 8
  },
  "tablas": [
    {
      "nombre": "tasa_observada_por_canal",
      "columnas": [
        "canal",
        "cuentas_expuestas",
        "tasa_observada",
        "shap_medio"
      ],
      "filas": [
        [
          "Branch",
          41250,
          0.052,
          -0.08
        ],
        [
          "Internet",
          18730,
          0.071,
          0.04
        ],
        [
          "TMK",
          9620,
          0.094,
          0.15
        ],
        [
          "Costco",
          12480,
          0.066,
          0.01
        ]
      ]
    }
  ],
  "frases_base": [
    "El canal es la variable 3 de 8 por importancia en el modelo de tasa de impago.",
    "Entre los canales mostrados, TMK tiene la tasa observada más alta (9.4%) y Branch la más baja (5.2%).",
    "Estas explicaciones describen cómo usa las variables el modelo; no establecen relaciones causales."
  ],
  "fragmentos": [],
  "versiones": {
    "modelo_s1": "s1/simulado",
    "modelo_s3": "s3/simulado",
    "datos_sha256": "simulado",
    "horizonte_mob": 6
  },
  "motivo": null,
  "simulado": true
}
```

### C3 — compare_groups

Archivo: `data/mock/evidencia_c3.json`

<!-- ejemplo: evidencia_c3 -->
```json
{
  "consulta_id": "C3",
  "estado": "ok",
  "funcion": "evidence.compare_groups",
  "parametros": {
    "dimension": "producto",
    "valor_a": "Costco",
    "valor_b": "Home Depot"
  },
  "cifras": {
    "tasa_a": 0.064,
    "tasa_b": 0.079,
    "diferencia": -0.015,
    "cuentas_expuestas_a": 15320,
    "cuentas_expuestas_b": 11870,
    "cohortes_maduras": 27,
    "horizonte_meses": 6
  },
  "tablas": [
    {
      "nombre": "comparacion_over30_mob6",
      "columnas": [
        "producto",
        "cuentas_expuestas",
        "cuentas_over30",
        "tasa_over30"
      ],
      "filas": [
        [
          "Costco",
          15320,
          980,
          0.064
        ],
        [
          "Home Depot",
          11870,
          938,
          0.079
        ]
      ]
    }
  ],
  "frases_base": [
    "En cohortes maduras, la tasa de Over30 a 6 meses es 6.4% en Costco y 7.9% en Home Depot.",
    "La diferencia observada es de -1.5 puntos porcentuales (Costco menos Home Depot).",
    "Es una comparación descriptiva; no indica que el producto cause la diferencia."
  ],
  "fragmentos": [],
  "versiones": {
    "datos_sha256": "simulado",
    "horizonte_mob": 6
  },
  "motivo": null,
  "simulado": true
}
```

### C4 — suggest_strategy

Archivo: `data/mock/evidencia_c4.json`

<!-- ejemplo: evidencia_c4 -->
```json
{
  "consulta_id": "C4",
  "estado": "ok",
  "funcion": "evidence.suggest_strategy",
  "parametros": {
    "perfil": {
      "producto": "Clasica",
      "canal": "Branch",
      "rango_edad": "03| Mature >30"
    }
  },
  "cifras": {
    "estrategia_sugerida": "r.Reca champ",
    "diferencia_tasa": 0.014,
    "intervalo_inferior": 0.004,
    "intervalo_superior": 0.024,
    "nivel_intervalo": 0.95,
    "metodos_coinciden": true,
    "cuentas_ventana": 3150
  },
  "tablas": [
    {
      "nombre": "tasa_por_estrategia",
      "columnas": [
        "estrategia",
        "metodo",
        "tasa_over30"
      ],
      "filas": [
        [
          "r.Reca champ",
          "observado",
          0.061
        ],
        [
          "DARIA 1.6",
          "observado",
          0.075
        ],
        [
          "r.Reca champ",
          "prediccion",
          0.063
        ],
        [
          "DARIA 1.6",
          "prediccion",
          0.076
        ]
      ]
    }
  ],
  "frases_base": [
    "Estrategia sugerida para este perfil: r.Reca champ.",
    "La tasa de Over30 de DARIA 1.6 es 1.4 puntos porcentuales mayor (intervalo de 95%: de 0.4 a 2.4).",
    "Los dos métodos coinciden en el signo. Es una sugerencia basada en la ventana de asignación aleatoria, no una decisión."
  ],
  "fragmentos": [],
  "versiones": {
    "modelo_s1": "s1/simulado",
    "modelo_s3": "s3/simulado",
    "datos_sha256": "simulado",
    "horizonte_mob": 6
  },
  "motivo": null,
  "simulado": true
}
```

### C5 — vintage_evolution

Archivo: `data/mock/evidencia_c5.json`

<!-- ejemplo: evidencia_c5 -->
```json
{
  "consulta_id": "C5",
  "estado": "ok",
  "funcion": "evidence.vintage_evolution",
  "parametros": {
    "vintage": 202506
  },
  "cifras": {
    "vintage": 202506,
    "horizonte_meses": 6,
    "tasa_over30_mob6": 0.058,
    "mob_maximo_observado": 15,
    "cuentas_mob0": 5210
  },
  "tablas": [
    {
      "nombre": "over30_por_mob",
      "columnas": [
        "mob",
        "cuentas_expuestas",
        "cuentas_over30",
        "tasa_over30"
      ],
      "filas": [
        [
          0,
          5210,
          0,
          0.0
        ],
        [
          1,
          4980,
          12,
          0.002
        ],
        [
          2,
          4870,
          58,
          0.012
        ],
        [
          3,
          4790,
          121,
          0.025
        ],
        [
          4,
          4720,
          176,
          0.037
        ],
        [
          5,
          4660,
          228,
          0.049
        ],
        [
          6,
          4600,
          267,
          0.058
        ]
      ]
    }
  ],
  "frases_base": [
    "En la vintage 202506, la tasa acumulada de Over30 llega a 5.8% en el MOB 6.",
    "Esta vintage se ha observado hasta el MOB 15."
  ],
  "fragmentos": [],
  "versiones": {
    "datos_sha256": "simulado",
    "horizonte_mob": 6
  },
  "motivo": null,
  "simulado": true
}
```

### C6 — summarize_batch

Archivo: `data/mock/evidencia_c6.json`

<!-- ejemplo: evidencia_c6 -->
```json
{
  "consulta_id": "C6",
  "estado": "ok",
  "funcion": "evidence.summarize_batch",
  "parametros": {
    "lote_id": "lote_simulado_001"
  },
  "cifras": {
    "perfiles_totales": 25,
    "perfiles_evaluados": 23,
    "perfiles_no_evaluables": 2,
    "monto_total_esperado": 41250000.0,
    "tasa_media_ponderada": 0.071,
    "perfiles_riesgo_critico": 1
  },
  "tablas": [
    {
      "nombre": "perfiles_por_nivel_de_riesgo",
      "columnas": [
        "nivel_riesgo",
        "perfiles"
      ],
      "filas": [
        [
          "Bajo",
          9
        ],
        [
          "Medio",
          8
        ],
        [
          "Alto",
          5
        ],
        [
          "Crítico",
          1
        ]
      ]
    }
  ],
  "frases_base": [
    "El lote tiene 25 perfiles; 23 se evaluaron y 2 no son evaluables.",
    "La tasa media ponderada de impago esperada es 7.1% y el monto total esperado es 41,250,000 pesos.",
    "1 perfil tiene riesgo Crítico y requiere revisión."
  ],
  "fragmentos": [],
  "versiones": {
    "modelo_s1": "s1/simulado",
    "modelo_s3": "s3/simulado",
    "datos_sha256": "simulado",
    "horizonte_mob": 6
  },
  "motivo": null,
  "simulado": true
}
```

### C7 — define_term

Archivo: `data/mock/evidencia_c7.json`

<!-- ejemplo: evidencia_c7 -->
```json
{
  "consulta_id": "C7",
  "estado": "ok",
  "funcion": "evidence.define_term",
  "parametros": {
    "termino": "GCL"
  },
  "cifras": {},
  "tablas": [],
  "frases_base": [],
  "fragmentos": [
    {
      "fuente": "data/knowledge/glosario.md",
      "titulo": "GCL (pérdida bruta)",
      "texto": "GCL es la pérdida bruta: la suma de los saldos de las cuentas que se dieron por perdidas, antes de descontar lo recuperado.",
      "puntaje": 0.83
    },
    {
      "fuente": "data/knowledge/glosario.md",
      "titulo": "NCL (pérdida neta)",
      "texto": "NCL es la pérdida neta: la pérdida bruta (GCL) menos lo recuperado.",
      "puntaje": 0.71
    }
  ],
  "versiones": {
    "base_documental": "simulado"
  },
  "motivo": null,
  "simulado": true
}
```

### Estado `no_evaluable` (C1 con DARIA 2.0)

Archivo: `data/mock/evidencia_no_evaluable.json`

<!-- ejemplo: evidencia_no_evaluable -->
```json
{
  "consulta_id": "C1",
  "estado": "no_evaluable",
  "funcion": "evidence.explain_profile",
  "parametros": {
    "perfil": {
      "estrategia": "DARIA 2.0",
      "segmento_estrategia": "Affluent",
      "producto": "Platinum",
      "canal": "Internet",
      "rango_edad": "03| Mature >30",
      "rango_score": "g.190-199"
    }
  },
  "cifras": {},
  "tablas": [],
  "frases_base": [],
  "fragmentos": [],
  "versiones": {
    "modelo_s1": "s1/simulado",
    "modelo_s3": "s3/simulado",
    "datos_sha256": "simulado",
    "horizonte_mob": 6
  },
  "motivo": "DARIA 2.0 solo tiene MOB 0 y 1: tiene línea asignada pero todavía no hay desempeño de riesgo a 6 meses.",
  "simulado": true
}
```

### Estado `sin_evidencia` (C3 sin cohortes maduras)

Archivo: `data/mock/evidencia_sin_evidencia.json`

<!-- ejemplo: evidencia_sin_evidencia -->
```json
{
  "consulta_id": "C3",
  "estado": "sin_evidencia",
  "funcion": "evidence.compare_groups",
  "parametros": {
    "dimension": "producto",
    "valor_a": "Teleton",
    "valor_b": "Beyond"
  },
  "cifras": {},
  "tablas": [],
  "frases_base": [],
  "fragmentos": [],
  "versiones": {
    "datos_sha256": "simulado",
    "horizonte_mob": 6
  },
  "motivo": "No hay cohortes maduras a 6 meses con cuentas expuestas para alguno de los dos grupos.",
  "simulado": true
}
```
