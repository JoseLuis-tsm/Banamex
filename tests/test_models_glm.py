"""Pruebas del GLM en JSON (src/models/glm.py): predicción, serialización y guardado como reserva."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import GammaRegressor, LogisticRegression

from src.models import registry
from src.models.glm import LinearGLM, save_reserve
from src.models.predict import load_s1, load_s3


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    return X


def test_glm_gamma_reproduce_a_scikit_learn(data):
    y = np.exp(1 + 0.3 * data["a"] - 0.2 * data["b"])
    sk = GammaRegressor(alpha=0.0, max_iter=1000).fit(data, y)
    glm = LinearGLM.from_sklearn(sk, list(data.columns), "log")
    assert glm.predict(data) == pytest.approx(sk.predict(data), rel=1e-10)


def test_glm_binomial_reproduce_a_scikit_learn(data):
    y = (data["a"] + 0.5 * data["c"] > 0).astype(int)
    sk = LogisticRegression(C=1.0, max_iter=1000).fit(data.to_numpy(), y)
    glm = LinearGLM.from_sklearn(sk, list(data.columns), "logit")
    assert glm.predict(data) == pytest.approx(sk.predict_proba(data.to_numpy())[:, 1], rel=1e-10)


def test_ida_y_vuelta_por_json_y_columnas_en_otro_orden(data):
    glm = LinearGLM("logit", -1.0, {"a": 0.5, "b": -0.25, "c": 0.0})
    back = LinearGLM.from_dict(json.loads(json.dumps(glm.to_dict())))
    assert back.predict(data[["c", "a", "b"]]) == pytest.approx(glm.predict(data))
    with pytest.raises(KeyError):
        glm.predict(data[["a", "b"]])
    with pytest.raises(ValueError):
        LinearGLM("identity", 0.0, {})


def test_reserva_no_toca_actual_json_y_se_carga_por_version(tmp_path):
    published = registry.publish(tmp_path, "s3", "s3-publicado")
    before = published.read_text(encoding="utf-8")
    glm = LinearGLM("logit", -2.0, {"x": 0.1})
    meta = {"datos_sha256": "abc", "hiperparametros": {"C": 1.0}}
    version, files = save_reserve(tmp_path, "s3", glm, meta)
    assert version.startswith("s3-") and all(f.exists() for f in files)
    assert published.read_text(encoding="utf-8") == before  # la reserva no se publica
    saved = registry.read_json(tmp_path / "s3" / version / "metadata.json")
    assert saved["tipo_modelo"] == "glm" and saved["rol"] == "reserva" and version in saved["como_activar"]
    assert save_reserve(tmp_path, "s3", glm, meta)[0] == version  # versión determinista


def test_vintage_list_por_conjunto():
    frame = pd.DataFrame({"VINTAGE": [202402, 202401, 202402, 202512], "CONJUNTO": ["a", "a", "a", "b"]})
    assert registry.vintage_list(frame, "a") == [202401, 202402]
    assert registry.vintage_list(frame) == [202401, 202402, 202512]


def test_cargar_una_version_inexistente_da_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_s1(tmp_path, "s1-no-existe")
    with pytest.raises(FileNotFoundError):
        load_s3(tmp_path, "s3-no-existe")
