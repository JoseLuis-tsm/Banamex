"""Carga de los modelos publicados y predicción por perfil (la usan S2, la API y el agente).

S1: el XGBoost predice log(línea); la línea es exp(predicción) · factor de retransformación
(guardado en metadata.json; 1.0 si no se usa).
S3: el XGBoost predice la tasa sin calibrar p; la tasa final pasa por la calibración de
Platt ajustada en validación: p_cal = σ(a · logit(p) + b), con a y b en metadata.json.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.engineering import build_features, feature_columns
from src.models.registry import current

MODEL_FILE = "modelo.json"
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    """logit(p) = log(p / (1 − p)), con p recortada a [EPS, 1 − EPS]."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """σ(z) = 1 / (1 + e^(−z))."""
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def platt_transform(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """Calibración de Platt: σ(a · logit(p) + b)."""
    return sigmoid(a * logit(p) + b)


def booster_predict(booster: xgb.Booster, features: pd.DataFrame) -> np.ndarray:
    """Predicción cruda del booster con las columnas en el orden de feature_columns()."""
    return booster.predict(xgb.DMatrix(features[feature_columns()], feature_names=feature_columns()))


class S1Model:
    """Modelo publicado de S1: línea promedio esperada por perfil."""

    def __init__(self, booster: xgb.Booster, metadata: Dict[str, Any]):
        self.booster = booster
        self.metadata = metadata
        self.smearing = float(metadata.get("retransformacion", {}).get("factor", 1.0))

    def predict_line(self, profiles: pd.DataFrame) -> np.ndarray:
        """Línea promedio esperada: exp(f(x)) · factor, con f el XGBoost sobre log(línea)."""
        return np.exp(booster_predict(self.booster, build_features(profiles))) * self.smearing


class S3Model:
    """Modelo publicado de S3: tasa calibrada de Over30 a 6 meses por perfil."""

    def __init__(self, booster: xgb.Booster, metadata: Dict[str, Any]):
        self.booster = booster
        self.metadata = metadata
        cal = metadata["calibracion"]
        self.a, self.b = float(cal["a"]), float(cal["b"])

    def predict_rate(self, profiles: pd.DataFrame) -> np.ndarray:
        """Tasa calibrada: σ(a · logit(p) + b), con p la salida del XGBoost."""
        return platt_transform(booster_predict(self.booster, build_features(profiles)), self.a, self.b)


def _load(models_dir: Path, output: str):
    found = current(models_dir, output)
    if found is None:
        raise FileNotFoundError(f"No hay modelo publicado de {output} (falta models/{output}/actual.json).")
    folder, metadata = found
    booster = xgb.Booster()
    booster.load_model(str(folder / MODEL_FILE))
    return booster, metadata


def load_s1(models_dir: Path) -> S1Model:
    """Carga el modelo de S1 publicado en models/s1/actual.json."""
    return S1Model(*_load(models_dir, "s1"))


def load_s3(models_dir: Path) -> S3Model:
    """Carga el modelo de S3 publicado en models/s3/actual.json."""
    return S3Model(*_load(models_dir, "s3"))
