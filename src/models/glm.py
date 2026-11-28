"""GLM guardado como JSON: los modelos de reserva de S1 (Gamma) y S3 (binomial).

Un GLM ajustado con scikit-learn se reduce a un intercepto y un coeficiente por variable,
así que se guarda en JSON (sin pickle) y se predice sin scikit-learn:
    η = β₀ + Σ βⱼ·xⱼ
    enlace "log"   (S1, GLM Gamma):    línea = exp(η)
    enlace "logit" (S3, GLM binomial): tasa  = σ(η) = 1 / (1 + e^(−η))
Código: β₀ = self.intercept, βⱼ = self.coef[columna], xⱼ = features[columna].
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.models import registry

LINKS = ("log", "logit")


class LinearGLM:
    """GLM con enlace log o logit, definido por su intercepto y sus coeficientes."""

    def __init__(self, link: str, intercept: float, coef: Dict[str, float]):
        if link not in LINKS:
            raise ValueError(f"Enlace desconocido: {link!r}; se esperaba uno de {LINKS}.")
        self.link = link
        self.intercept = float(intercept)
        self.coef = {k: float(v) for k, v in coef.items()}

    @classmethod
    def from_sklearn(cls, model, columns: List[str], link: str) -> "LinearGLM":
        """Toma β₀ y βⱼ de un GammaRegressor o LogisticRegression ya ajustado.

        LogisticRegression guarda coef_ con forma (1, p) e intercept_ con forma (1,);
        GammaRegressor, coef_ con forma (p,) e intercept_ escalar. np.ravel unifica ambos.
        """
        coef = np.ravel(model.coef_)
        if len(coef) != len(columns):
            raise ValueError(f"El modelo tiene {len(coef)} coeficientes y se dieron {len(columns)} columnas.")
        return cls(link, float(np.ravel(model.intercept_)[0]), dict(zip(columns, coef)))

    def linear_predictor(self, features: pd.DataFrame) -> np.ndarray:
        """η = β₀ + Σ βⱼ·xⱼ, con las columnas en el orden en que se guardaron."""
        columns = list(self.coef)
        missing = [c for c in columns if c not in features.columns]
        if missing:
            raise KeyError(f"Faltan variables del GLM: {missing[:5]}")
        beta = np.array([self.coef[c] for c in columns])
        return self.intercept + features[columns].to_numpy(dtype=float) @ beta

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """exp(η) con enlace log; σ(η) con enlace logit."""
        eta = self.linear_predictor(features)
        return np.exp(eta) if self.link == "log" else 1 / (1 + np.exp(-eta))

    def to_dict(self) -> Dict:
        """Diccionario serializable a JSON."""
        return {"tipo": "glm", "enlace": self.link, "intercepto": self.intercept, "coeficientes": self.coef}

    @classmethod
    def from_dict(cls, data: Dict) -> "LinearGLM":
        """Reconstruye el modelo desde to_dict()."""
        return cls(data["enlace"], data["intercepto"], data["coeficientes"])


# ===== Guardado como modelo de reserva =====
RESERVE_ROLE = "reserva"


def activation_note(output: str, version: str) -> str:
    """Texto de cómo activar una reserva (se guarda en su metadata.json)."""
    return (f"Modelo de reserva, no publicado. Para usarlo, apunta models/{output}/actual.json a "
            f"{version} (o cárgalo con load_{output}(models_dir, '{version}')). La explicabilidad "
            "de la Fase 3 está hecha para los modelos publicados y no aplica a esta reserva.")


def save_reserve(models_dir: Path, output: str, model: LinearGLM, metadata: Dict[str, Any]) -> Tuple[str, List[Path]]:
    """Guarda un GLM como reserva en models/<salida>/<versión>/ sin publicarlo.

    La versión es determinista (registry.version_id) sobre el tipo, los datos, las variables y
    los hiperparámetros: repetir la corrida reescribe la misma carpeta. Nunca toca actual.json.
    `metadata` debe traer "datos_sha256" e "hiperparametros".
    Devuelve: (versión, [modelo.json, metadata.json]).
    """
    version = registry.version_id(output, {"tipo": "glm", "datos": metadata["datos_sha256"],
                                           "variables": list(model.coef), "hiperparametros": metadata["hiperparametros"]})
    folder = registry.version_dir(models_dir, output, version)
    model_path = registry.write_json(folder / "modelo.json", model.to_dict())
    full = {"tipo_modelo": "glm", "rol": RESERVE_ROLE, "como_activar": activation_note(output, version),
            "enlace": model.link, **metadata, "archivos": {"modelo": "modelo.json"}}
    return version, [model_path, registry.save_metadata(models_dir, output, version, full)]
