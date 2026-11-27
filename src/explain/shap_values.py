"""Valores SHAP de S3 agrupados por variable original (CLAUDE.md §1, Fase 3C).

Cálculo: TreeSHAP exacto de XGBoost (Booster.predict(..., pred_contribs=True)). Es el mismo
algoritmo que shap.TreeExplainer; se usa el nativo porque shap 0.49.1 no lee el formato de
base_score de XGBoost 3.x. Los valores están en **escala logit** del modelo **sin calibrar**:

    logit(p_sin_calibrar) = valor_base + Σ_j φ_j

con φ_j la contribución de la columna j. Las columnas one-hot de una misma variable
(por ejemplo, las 12 de PRODUCT) se **suman** para obtener la contribución de la variable
original; la suma conserva la aditividad.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.engineering import AGE_FEATURE, FAMILY_FEATURE, MONTH_FEATURE, SCORE_FEATURE, ONE_HOT_SOURCES, feature_columns

# Nombre en español de cada variable original, en el orden en que se muestran.
VARIABLE_LABELS: Dict[str, str] = {
    "STRATEGY": "Estrategia",
    "STRATEGY_SEGMENT": "Segmento",
    "PRODUCT": "Producto",
    "CHANNEL": "Canal",
    "AGE_TAG": "Edad",
    "SCORE_TAG": "Score",
    "FAMILY": "Familia de estrategia",
    "BOOKING_MONTH": "Mes de booking",
}
VARIABLES: List[str] = list(VARIABLE_LABELS)

NOTE = "Estas explicaciones describen cómo usa las variables el modelo; no establecen relaciones causales."
SCALE_NOTE = "Valores SHAP en escala logit del modelo sin calibrar: indican dirección y tamaño relativo, no puntos porcentuales."


def feature_to_variable() -> Dict[str, str]:
    """Mapa columna del modelo → variable original (one-hot y numéricas)."""
    mapping = {}
    for source in ONE_HOT_SOURCES:
        for col in feature_columns():
            if col.startswith(source + "__"):
                mapping[col] = source
    mapping.update({AGE_FEATURE: "AGE_TAG", SCORE_FEATURE: "SCORE_TAG", FAMILY_FEATURE: "FAMILY", MONTH_FEATURE: "BOOKING_MONTH"})
    return mapping


def shap_by_column(booster: xgb.Booster, features: pd.DataFrame) -> pd.DataFrame:
    """Contribuciones TreeSHAP por columna del modelo, más la columna VALOR_BASE.

    Devuelve DataFrame con feature_columns() + ["VALOR_BASE"], mismo índice que features.
    """
    d = xgb.DMatrix(features[feature_columns()], feature_names=feature_columns())
    contribs = booster.predict(d, pred_contribs=True)
    out = pd.DataFrame(contribs[:, :-1], columns=feature_columns(), index=features.index)
    out["VALOR_BASE"] = contribs[:, -1]
    return out


def group_by_variable(by_column: pd.DataFrame) -> pd.DataFrame:
    """Suma las contribuciones por variable original. Conserva VALOR_BASE."""
    mapping = feature_to_variable()
    grouped = pd.DataFrame(index=by_column.index)
    for var in VARIABLES:
        cols = [c for c, v in mapping.items() if v == var]
        grouped[var] = by_column[cols].sum(axis=1)
    grouped["VALOR_BASE"] = by_column["VALOR_BASE"]
    return grouped


def shap_by_variable(booster: xgb.Booster, features: pd.DataFrame) -> pd.DataFrame:
    """Atajo: contribuciones agrupadas por variable original + VALOR_BASE + SALIDA_LOGIT."""
    grouped = group_by_variable(shap_by_column(booster, features))
    d = xgb.DMatrix(features[feature_columns()], feature_names=feature_columns())
    grouped["SALIDA_LOGIT"] = booster.predict(d, output_margin=True)
    return grouped


def variable_values(dims: pd.DataFrame) -> pd.DataFrame:
    """Valor que toma cada variable original en cada fila (para etiquetas y frases)."""
    from src.data import catalog as cat

    out = pd.DataFrame(index=dims.index)
    for var in ("STRATEGY", "STRATEGY_SEGMENT", "PRODUCT", "CHANNEL", "AGE_TAG", "SCORE_TAG"):
        out[var] = dims[var].astype(str)
    out["FAMILY"] = dims["STRATEGY"].astype(str).map(cat.STRATEGY_FAMILY)
    out["BOOKING_MONTH"] = dims["VINTAGE"].astype(int).map(cat.booking_month).astype(str)
    return out


def global_importance(grouped: pd.DataFrame, weights: np.ndarray = None) -> pd.DataFrame:
    """Importancia global: media (ponderada) del valor absoluto SHAP por variable original.

    Fórmula: I_v = Σ_i w_i·|φ_iv| / Σ_i w_i, con w_i el peso de la fila (ENRACCTS) o 1.
    """
    w = np.ones(len(grouped)) if weights is None else np.asarray(weights, dtype=float)
    rows = [{"variable": v, "etiqueta": VARIABLE_LABELS[v], "importancia": float(np.sum(w * grouped[v].abs()) / np.sum(w))} for v in VARIABLES]
    return pd.DataFrame(rows).sort_values("importancia", ascending=False).reset_index(drop=True)


def additivity_error(grouped: pd.DataFrame) -> float:
    """Máximo |valor_base + Σ φ − salida_logit| sobre las filas."""
    return float(np.max(np.abs(grouped["VALOR_BASE"] + grouped[VARIABLES].sum(axis=1) - grouped["SALIDA_LOGIT"])))


SHAP_PREFIX = "SHAP_"


def to_table(dims: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    """Une dimensiones y contribuciones; las contribuciones van como SHAP_<variable>."""
    return pd.concat([dims.reset_index(drop=True), grouped.reset_index(drop=True).rename(columns={v: SHAP_PREFIX + v for v in VARIABLES})], axis=1)


def from_table(table: pd.DataFrame) -> pd.DataFrame:
    """Recupera las contribuciones agrupadas (columnas = VARIABLES, VALOR_BASE, SALIDA_LOGIT)."""
    out = table[[SHAP_PREFIX + v for v in VARIABLES] + ["VALOR_BASE", "SALIDA_LOGIT"]].copy()
    out.columns = VARIABLES + ["VALOR_BASE", "SALIDA_LOGIT"]
    return out
