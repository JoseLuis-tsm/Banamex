"""Carga de los modelos publicados y predicción por perfil (la usan S2, la API y el agente).

S1 puede ser de dos tipos (metadata.json → "tipo_modelo"):
  - "base_simple": línea media ponderada por producto × score (Σ CREDITLINESAMTS / Σ ACCOUNTS),
    con respaldo a solo producto y a global. Es el publicado cuando el XGBoost no supera
    su criterio (decisión de Luis en la Fase 2).
  - "xgboost": el XGBoost predice log(línea); la línea es exp(predicción) · factor de
    retransformación (guardado en metadata.json; 1.0 si no se usa).
  - "glm": GLM Gamma con enlace log (reserva): línea = exp(β₀ + Σ βⱼ·xⱼ).
S3 puede ser de dos tipos:
  - "xgboost" (publicado): predice la tasa sin calibrar p; la tasa final pasa por la
    calibración de Platt ajustada en validación: p_cal = σ(a · logit(p) + b), con a y b en
    metadata.json.
  - "glm": GLM binomial (reserva): tasa = σ(β₀ + Σ βⱼ·xⱼ), sin calibración aparte.
Los modelos de reserva no se publican; se activan apuntando actual.json a su versión, o
se cargan directamente con load_s1(models_dir, version) / load_s3(models_dir, version).
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.engineering import build_features, feature_columns
from src.models.baselines import RatioBaseline
from src.models.glm import LinearGLM
from src.models.registry import current, read_json

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
    """Modelo de S1: línea promedio esperada por perfil (base simple, XGBoost o GLM)."""

    def __init__(self, model: Any, metadata: Dict[str, Any]):
        self.model = model
        self.metadata = metadata
        self.kind = metadata.get("tipo_modelo", "xgboost")
        self.smearing = float(metadata.get("retransformacion", {}).get("factor", 1.0))

    def predict_line(self, profiles: pd.DataFrame) -> np.ndarray:
        """Línea promedio esperada por cuenta.

        base_simple: Σ CREDITLINESAMTS / Σ ACCOUNTS del grupo (producto × score, o respaldo).
        xgboost: exp(f(x)) · factor, con f el XGBoost sobre log(línea).
        glm: exp(β₀ + Σ βⱼ·xⱼ).
        """
        if self.kind == "base_simple":
            build_features(profiles)  # valida que el perfil esté en el catálogo
            return self.model.predict(profiles)
        if self.kind == "glm":
            return self.model.predict(build_features(profiles))
        return np.exp(booster_predict(self.model, build_features(profiles))) * self.smearing


class S3Model:
    """Modelo de S3: tasa de Over30 a 6 meses por perfil (XGBoost calibrado o GLM)."""

    def __init__(self, model: Any, metadata: Dict[str, Any]):
        self.model = model
        self.metadata = metadata
        self.kind = metadata.get("tipo_modelo", "xgboost")
        if self.kind == "xgboost":
            cal = metadata["calibracion"]
            self.a, self.b = float(cal["a"]), float(cal["b"])

    @property
    def booster(self) -> xgb.Booster:
        """El booster del XGBoost (lo usa la explicabilidad); solo existe en el tipo xgboost."""
        if self.kind != "xgboost":
            raise TypeError(f"El modelo de S3 cargado es de tipo {self.kind!r}, no un XGBoost.")
        return self.model

    def predict_rate(self, profiles: pd.DataFrame) -> np.ndarray:
        """xgboost: σ(a · logit(p) + b), con p la salida del XGBoost. glm: σ(β₀ + Σ βⱼ·xⱼ)."""
        features = build_features(profiles)
        if self.kind == "glm":
            return self.model.predict(features)
        return platt_transform(booster_predict(self.model, features), self.a, self.b)


def _load(models_dir: Path, output: str, version: Optional[str] = None):
    """Modelo y metadata de la versión publicada (actual.json) o de la versión indicada."""
    if version is None:
        found = current(models_dir, output)
        if found is None:
            raise FileNotFoundError(f"No hay modelo publicado de {output} (falta models/{output}/actual.json).")
        folder, metadata = found
    else:
        folder = Path(models_dir) / output / version
        if not (folder / "metadata.json").exists():
            raise FileNotFoundError(f"No existe la versión {version} (falta {folder / 'metadata.json'}).")
        metadata = read_json(folder / "metadata.json")
    if not (folder / MODEL_FILE).exists():
        raise FileNotFoundError(f"Falta {folder / MODEL_FILE}: el binario no va a git; regenéralo con su script de la Fase 2.")
    kind = metadata.get("tipo_modelo")
    if kind == "base_simple":
        return RatioBaseline.from_dict(read_json(folder / MODEL_FILE)), metadata
    if kind == "glm":
        return LinearGLM.from_dict(read_json(folder / MODEL_FILE)), metadata
    booster = xgb.Booster()
    booster.load_model(str(folder / MODEL_FILE))
    return booster, metadata


def load_s1(models_dir: Path, version: Optional[str] = None) -> S1Model:
    """Carga el S1 publicado (models/s1/actual.json) o, con `version`, esa versión (p. ej. la reserva)."""
    return S1Model(*_load(models_dir, "s1", version))


def load_s3(models_dir: Path, version: Optional[str] = None) -> S3Model:
    """Carga el S3 publicado (models/s3/actual.json) o, con `version`, esa versión (p. ej. la reserva)."""
    return S3Model(*_load(models_dir, "s3", version))
