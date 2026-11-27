"""Líneas base simples: razón de sumas por grupo, con respaldo jerárquico.

S1: línea media ponderada por producto y score = Σ CREDITLINESAMTS / Σ ACCOUNTS en el grupo.
S3: tasa ponderada por score y producto        = Σ OVER30ACCTS / Σ ENRACCTS en el grupo.

Si un grupo no se vio en entrenamiento (por ejemplo, un rango de score DARIA, que solo
aparece en validación y prueba), se usa el siguiente nivel: solo producto y, al final,
el valor global de entrenamiento.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class RatioBaseline:
    """Predice Σ numerador / Σ denominador del grupo más específico visto en entrenamiento."""

    def __init__(self, levels: List[List[str]], numerator: str, denominator: str):
        """levels: del más específico al más general; [] es el global."""
        self.levels = levels
        self.numerator = numerator
        self.denominator = denominator
        self.tables: List[Tuple[List[str], Dict]] = []

    def fit(self, df: pd.DataFrame) -> "RatioBaseline":
        """Calcula la razón de sumas por cada nivel con los datos de entrenamiento."""
        self.tables = []
        for keys in self.levels:
            if keys:
                g = df.groupby(keys, observed=True)[[self.numerator, self.denominator]].sum()
                ratio = (g[self.numerator] / g[self.denominator]).to_dict()
            else:
                ratio = {(): df[self.numerator].sum() / df[self.denominator].sum()}
            self.tables.append((keys, ratio))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicción por fila con respaldo jerárquico; también devuelve qué nivel se usó en self.last_levels."""
        pred = np.full(len(df), np.nan)
        used = np.full(len(df), -1)
        for level, (keys, ratio) in enumerate(self.tables):
            if keys:
                idx = pd.MultiIndex.from_frame(df[keys].astype(str)) if len(keys) > 1 else df[keys[0]].astype(str)
                values = np.array([ratio.get(k if len(keys) > 1 else k, np.nan) for k in idx], dtype=float)
            else:
                values = np.full(len(df), ratio[()])
            fill = np.isnan(pred) & ~np.isnan(values)
            pred[fill] = values[fill]
            used[fill] = level
        self.last_levels = used
        return pred
