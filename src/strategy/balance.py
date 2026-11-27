"""Diagnóstico de balance entre dos estrategias (mezcla de perfiles).

Se usa para comprobar que, en la ventana de asignación aleatoria, r.Reca champ y DARIA
recibieron perfiles parecidos en las dimensiones compartidas (edad, producto, canal y
posición de score). Medida: diferencia estandarizada de medias (SMD); |SMD| > 0.10 es
la convención habitual para marcar desbalance.
"""

from typing import List

import numpy as np
import pandas as pd

SMD_THRESHOLD = 0.10


def smd_proportion(p_a: float, p_b: float) -> float:
    """SMD de una proporción (variable indicadora de una categoría).

    Fórmula: SMD = (p_a − p_b) / √[(p_a(1 − p_a) + p_b(1 − p_b)) / 2],
    con p_a y p_b la participación de la categoría en cada estrategia (entre 0 y 1).
    Devuelve 0 si ambas varianzas son 0 (la categoría está en 0% o 100% en las dos).
    """
    pooled = np.sqrt((p_a * (1 - p_a) + p_b * (1 - p_b)) / 2)
    return 0.0 if pooled == 0 else float((p_a - p_b) / pooled)


def weighted_mean_sd(values: pd.Series, weights: pd.Series) -> tuple:
    """Media y desviación estándar ponderadas: μ = Σwx/Σw, σ² = Σw(x − μ)²/Σw."""
    w = weights.to_numpy(dtype=float)
    x = values.to_numpy(dtype=float)
    mean = np.sum(w * x) / np.sum(w)
    return float(mean), float(np.sqrt(np.sum(w * (x - mean) ** 2) / np.sum(w)))


def smd_continuous(mean_a: float, sd_a: float, mean_b: float, sd_b: float) -> float:
    """SMD de una variable continua: (μ_a − μ_b) / √[(σ_a² + σ_b²) / 2]."""
    pooled = np.sqrt((sd_a ** 2 + sd_b ** 2) / 2)
    return 0.0 if pooled == 0 else float((mean_a - mean_b) / pooled)


def categorical_balance(df: pd.DataFrame, column: str, group: str, a: str, b: str, weight: str = "ACCOUNTS") -> pd.DataFrame:
    """Participación ponderada de cada categoría de `column` en los grupos `a` y `b`, con su SMD.

    Devuelve columnas: variable, categoria, participacion_a, participacion_b,
    diferencia_pp (puntos porcentuales, a − b), smd y desbalance (|smd| > SMD_THRESHOLD).
    """
    t = df.groupby([group, column], observed=True)[weight].sum().unstack(group, fill_value=0)
    for g in (a, b):
        if g not in t.columns:
            t[g] = 0
    shares = t[[a, b]] / t[[a, b]].sum()
    rows: List[dict] = []
    for category, (p_a, p_b) in shares.iterrows():
        smd = smd_proportion(p_a, p_b)
        rows.append({
            "variable": column, "categoria": str(category),
            "participacion_a": p_a, "participacion_b": p_b,
            "diferencia_pp": 100 * (p_a - p_b), "smd": smd, "desbalance": abs(smd) > SMD_THRESHOLD,
        })
    return pd.DataFrame(rows)


def continuous_balance(df: pd.DataFrame, column: str, group: str, a: str, b: str, weight: str = "ACCOUNTS") -> pd.DataFrame:
    """Media ponderada de `column` en `a` y `b`, con su SMD (una fila)."""
    mean_a, sd_a = weighted_mean_sd(df.loc[df[group] == a, column], df.loc[df[group] == a, weight])
    mean_b, sd_b = weighted_mean_sd(df.loc[df[group] == b, column], df.loc[df[group] == b, weight])
    smd = smd_continuous(mean_a, sd_a, mean_b, sd_b)
    return pd.DataFrame([{
        "variable": column, "categoria": "media",
        "participacion_a": mean_a, "participacion_b": mean_b,
        "diferencia_pp": np.nan, "smd": smd, "desbalance": abs(smd) > SMD_THRESHOLD,
    }])
