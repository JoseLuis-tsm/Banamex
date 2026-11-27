"""Pruebas de carga y predicción de modelos publicados (src/models/predict.py)."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.features.engineering import build_features, feature_columns
from src.models import registry
from src.models.predict import MODEL_FILE, load_s1, load_s3, platt_transform


def _perfiles(n=40):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "VINTAGE": rng.choice([202401, 202506, 202512], n), "STRATEGY": "r.Reca champ",
        "STRATEGY_SEGMENT": "Costco", "PRODUCT": rng.choice(["Costco", "Clasica"], n), "CHANNEL": "Costco",
        "AGE_TAG": "02| Young 25-30", "SCORE_TAG": rng.choice(["<130", "151-160", "190+"], n),
    })


def _publicar(tmp_path, output, label, objective, metadata):
    df = _perfiles()
    d = xgb.DMatrix(build_features(df), label=label(len(df)), feature_names=feature_columns())
    booster = xgb.train({"objective": objective, "max_depth": 2, "seed": 42}, d, num_boost_round=5)
    version = registry.version_id(output, {"t": 1})
    booster.save_model(str(registry.version_dir(tmp_path, output, version) / MODEL_FILE))
    registry.save_metadata(tmp_path, output, version, metadata)
    registry.publish(tmp_path, output, version)
    return df, booster


def test_s1_aplica_exp_y_factor(tmp_path):
    df, booster = _publicar(tmp_path, "s1", lambda n: np.log(np.full(n, 40000.0)), "reg:squarederror",
                            {"retransformacion": {"factor": 1.1}})
    raw = booster.predict(xgb.DMatrix(build_features(df), feature_names=feature_columns()))
    assert load_s1(tmp_path).predict_line(df) == pytest.approx(np.exp(raw) * 1.1, rel=1e-6)


def test_s3_aplica_platt(tmp_path):
    df, booster = _publicar(tmp_path, "s3", lambda n: np.full(n, 0.08), "binary:logistic",
                            {"calibracion": {"a": 0.9, "b": -0.1}})
    raw = booster.predict(xgb.DMatrix(build_features(df), feature_names=feature_columns()))
    pred = load_s3(tmp_path).predict_rate(df)
    assert pred == pytest.approx(platt_transform(raw, 0.9, -0.1), rel=1e-6)
    assert ((pred > 0) & (pred < 1)).all()


def test_sin_modelo_publicado_hay_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="actual.json"):
        load_s1(tmp_path)
