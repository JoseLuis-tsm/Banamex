"""Explicación de S1: descomposición exacta en factores de la línea base publicada.

El S1 publicado es la línea base simple (RatioBaseline): la línea promedio del grupo
producto × score visto en entrenamiento, con respaldo a solo producto y al global. Por eso
se explica exactamente, sin aproximación:

    línea = promedio global × factor del producto × factor del score dentro del producto

    factor del producto = r_p / g        (r_p: línea promedio del producto; g: global)
    factor del score    = r_ps / r_p     (r_ps: línea promedio del producto y score)

En logaritmo es aditiva, como SHAP:  log(línea) = log g + log(r_p/g) + log(r_ps/r_p).
Si el par producto × score no se vio, el factor del score es 1 (se usó el nivel producto);
si tampoco el producto, ambos factores son 1 (se usó el global). Las demás variables tienen
contribución 0: el S1 publicado no las usa.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.explain.shap_values import NOTE, VARIABLE_LABELS, VARIABLES
from src.models.baselines import RatioBaseline

S1_VARIABLES: List[str] = ["PRODUCT", "SCORE_TAG"]
S1_SCALE_NOTE = "Contribuciones en logaritmo de la línea: exp(contribución) es el factor multiplicativo sobre el promedio general."
SALIDA_LOG = "SALIDA_LOG"
MIN_FACTOR_GAP = 5e-4  # |log factor| menor que esto se redondea a factor 1.00 y no se lista

UP_TEMPLATE = "{label} «{value}»: {detail} es {factor:.2f} veces {reference} (línea estimada más alta para este perfil)."
DOWN_TEMPLATE = "{label} «{value}»: {detail} es {factor:.2f} veces {reference} (línea estimada más baja para este perfil)."


def _check_structure(model: RatioBaseline) -> None:
    levels = [keys for keys, _ in model.tables]
    if levels != [["PRODUCT", "SCORE_TAG"], ["PRODUCT"], []]:
        raise ValueError(f"La descomposición supone niveles [[PRODUCT, SCORE_TAG], [PRODUCT], []]; el modelo tiene {levels}.")


def decompose(model: RatioBaseline, profiles: pd.DataFrame) -> pd.DataFrame:
    """Contribuciones en log por variable original, VALOR_BASE = log g y SALIDA_LOG = log(línea).

    Devuelve un DataFrame con VARIABLES (las que S1 no usa valen 0), VALOR_BASE, SALIDA_LOG y
    NIVEL (0 = producto × score, 1 = solo producto, 2 = global), mismo índice que profiles.
    """
    _check_structure(model)
    (_, ps_table), (_, p_table), (_, g_table) = model.tables
    g = float(g_table[()])
    product = profiles["PRODUCT"].astype(str)
    score = profiles["SCORE_TAG"].astype(str)
    r_p = product.map(p_table).astype(float)
    r_ps = pd.Series([ps_table.get((p, s), np.nan) for p, s in zip(product, score)], index=profiles.index, dtype=float)
    # Respaldo: sin r_p el factor del producto es 1; sin r_ps el del score es 1.
    r_p_used = r_p.fillna(g)
    r_ps_used = r_ps.fillna(r_p_used)
    out = pd.DataFrame(0.0, index=profiles.index, columns=VARIABLES)
    out["PRODUCT"] = np.log(r_p_used / g)          # log(r_p / g)
    out["SCORE_TAG"] = np.log(r_ps_used / r_p_used)  # log(r_ps / r_p)
    out["VALOR_BASE"] = np.log(g)
    out[SALIDA_LOG] = np.log(model.predict(profiles))
    out["NIVEL"] = model.last_levels
    return out


def additivity_error(decomposition: pd.DataFrame) -> float:
    """Máximo |log g + Σ contribuciones − log(línea)| sobre las filas."""
    d = decomposition
    return float(np.max(np.abs(d["VALOR_BASE"] + d[VARIABLES].sum(axis=1) - d[SALIDA_LOG])))


def reading_phrases(contrib: pd.Series, values: pd.Series) -> Dict[str, object]:
    """Frases de lectura de S1 para un perfil: cada factor distinto de 1, con su valor.

    Parámetros:
        contrib: contribución en log por variable (índice = VARIABLES).
        values: valor de cada variable en el perfil (mismo índice).
    Devuelve: {"suben": [...], "bajan": [...], "nota": NOTE}.
    """
    details = {
        "PRODUCT": ("la línea promedio del producto", "el promedio general"),
        "SCORE_TAG": (f"dentro de {values['PRODUCT']}, la línea promedio de este rango", "la del producto"),
    }
    up, down = [], []
    for var in S1_VARIABLES:
        c = float(contrib[var])
        if abs(c) <= MIN_FACTOR_GAP:
            continue
        detail, reference = details[var]
        template = UP_TEMPLATE if c > 0 else DOWN_TEMPLATE
        text = template.format(label=VARIABLE_LABELS[var], value=values[var], detail=detail, factor=float(np.exp(c)), reference=reference)
        (up if c > 0 else down).append(text)
    return {"suben": up, "bajan": down, "nota": NOTE}


def as_text(phrases: Dict[str, object]) -> str:
    """Las frases de S1 como texto plano: suben, bajan, nota."""
    lines = ["Factores que suben la línea estimada:"] + [f"- {p}" for p in phrases["suben"] or ["(ninguno)"]]
    lines += ["Factores que bajan la línea estimada:"] + [f"- {p}" for p in phrases["bajan"] or ["(ninguno)"]]
    return "\n".join(lines + [phrases["nota"]])
