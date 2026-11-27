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

    def to_dict(self) -> Dict:
        """Representación JSON: niveles y, por nivel, filas {clave..., valor}."""
        levels = []
        for keys, ratio in self.tables:
            rows = [{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "valor": float(v)} for k, v in ratio.items()] if keys else [{"valor": float(ratio[()])}]
            levels.append({"claves": keys, "filas": rows})
        return {"tipo": "RatioBaseline", "numerador": self.numerator, "denominador": self.denominator, "niveles": levels}

    @classmethod
    def from_dict(cls, data: Dict) -> "RatioBaseline":
        """Reconstruye la línea base desde to_dict()."""
        model = cls([lvl["claves"] for lvl in data["niveles"]], data["numerador"], data["denominador"])
        for lvl in data["niveles"]:
            keys = lvl["claves"]
            if not keys:
                ratio = {(): lvl["filas"][0]["valor"]}
            elif len(keys) == 1:
                ratio = {r[keys[0]]: r["valor"] for r in lvl["filas"]}
            else:
                ratio = {tuple(r[k] for k in keys): r["valor"] for r in lvl["filas"]}
            model.tables.append((keys, ratio))
        return model
