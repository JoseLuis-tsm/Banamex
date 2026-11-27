"""Frases de lectura de una explicación SHAP, por plantilla (sin modelo de lenguaje).

Para un perfil: los tres factores que más suben y los tres que más bajan la tasa estimada,
en español, sin lenguaje causal, y siempre con la nota fija de no causalidad.
"""

import re
from typing import Dict, List

import pandas as pd

from src.explain.shap_values import NOTE, VARIABLE_LABELS

TOP_N = 3
MIN_ABS = 5e-4  # contribuciones que se redondean a 0.000 no se listan como factor

# Palabras que implican causalidad y no deben aparecer en las frases.
CAUSAL_WORDS = ["causa", "provoca", "ocasiona", "genera", "produce", "debido a", "porque", "por culpa", "hace que", "origina"]

UP_TEMPLATE = "{label} «{value}»: el modelo lo asocia con una tasa estimada más alta para este perfil (contribución {contrib:+.3f})."
DOWN_TEMPLATE = "{label} «{value}»: el modelo lo asocia con una tasa estimada más baja para este perfil (contribución {contrib:+.3f})."


def top_factors(contrib: pd.Series, n: int = TOP_N) -> Dict[str, List[str]]:
    """Variables con mayor contribución positiva y negativa (hasta n de cada una)."""
    c = contrib[contrib.abs() > MIN_ABS]
    up = c[c > 0].sort_values(ascending=False).head(n).index.tolist()
    down = c[c < 0].sort_values().head(n).index.tolist()
    return {"suben": up, "bajan": down}


def reading_phrases(contrib: pd.Series, values: pd.Series, n: int = TOP_N) -> Dict[str, object]:
    """Frases de lectura de un perfil.

    Parámetros:
        contrib: contribución SHAP por variable original (índice = VARIABLES).
        values: valor que toma cada variable en el perfil (mismo índice).

    Devuelve:
        {"suben": [frases], "bajan": [frases], "nota": NOTE}.
    """
    f = top_factors(contrib, n)
    up = [UP_TEMPLATE.format(label=VARIABLE_LABELS[v], value=values[v], contrib=contrib[v]) for v in f["suben"]]
    down = [DOWN_TEMPLATE.format(label=VARIABLE_LABELS[v], value=values[v], contrib=contrib[v]) for v in f["bajan"]]
    return {"suben": up, "bajan": down, "nota": NOTE}


def causal_terms(text: str) -> List[str]:
    """Palabras causales que aparecen en el texto (como palabra completa, sin distinguir mayúsculas)."""
    return [w for w in CAUSAL_WORDS if re.search(r"\b" + re.escape(w) + r"\b", text, flags=re.IGNORECASE)]


def as_text(phrases: Dict[str, object]) -> str:
    """Las frases como texto plano, en el orden: suben, bajan, nota."""
    lines = ["Factores que más suben la tasa estimada:"] + [f"- {p}" for p in phrases["suben"] or ["(ninguno)"]]
    lines += ["Factores que más bajan la tasa estimada:"] + [f"- {p}" for p in phrases["bajan"] or ["(ninguno)"]]
    return "\n".join(lines + [phrases["nota"]])
