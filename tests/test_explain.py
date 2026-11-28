"""Pruebas de explicabilidad (src/explain/): agrupación SHAP, aditividad y frases."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.explain.phrases import CAUSAL_WORDS, as_text, causal_terms, reading_phrases, top_factors
from src.explain.shap_values import (
    NOTE, VARIABLES, additivity_error, feature_to_variable, from_table, global_importance, shap_by_column,
    shap_by_variable, to_table,
)
from src.features.engineering import build_features, feature_columns


def _modelo():
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "VINTAGE": rng.choice([202401, 202506], n), "STRATEGY": "r.Reca champ", "STRATEGY_SEGMENT": rng.choice(["Costco", "Brands-Payroll-OnUs"], n),
        "PRODUCT": rng.choice(["Costco", "Clasica", "Oro"], n), "CHANNEL": rng.choice(["Costco", "Branch"], n),
        "AGE_TAG": rng.choice(["01| Young < 25", "03| Mature >30"], n), "SCORE_TAG": rng.choice(["<130", "151-160", "190+"], n),
    })
    x = build_features(df)
    y = np.clip(0.15 - 0.1 * x["SCORE_POSICION"] + 0.03 * (df["PRODUCT"] == "Oro") + rng.normal(0, 0.01, n), 0.01, 0.9)
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 2, "seed": 42}, xgb.DMatrix(x, label=y, feature_names=feature_columns()), 30)
    return df, x, booster


def test_cada_columna_del_modelo_pertenece_a_una_variable():
    mapping = feature_to_variable()
    assert set(mapping) == set(feature_columns())
    assert set(mapping.values()) == set(VARIABLES)


def test_agrupar_conserva_la_aditividad():
    df, x, booster = _modelo()
    by_col = shap_by_column(booster, x)
    grouped = shap_by_variable(booster, x)
    assert np.allclose(grouped[VARIABLES].sum(axis=1), by_col[feature_columns()].sum(axis=1), atol=1e-6)
    assert additivity_error(grouped) < 1e-4


def test_tabla_ida_y_vuelta_e_importancia():
    df, x, booster = _modelo()
    grouped = shap_by_variable(booster, x)
    back = from_table(to_table(df, grouped))
    assert np.allclose(back.to_numpy(), grouped.to_numpy())
    imp = global_importance(grouped)
    assert imp.iloc[0]["variable"] == "SCORE_TAG" and (imp["importancia"] >= 0).all()


def test_frases_tres_y_tres_sin_lenguaje_causal():
    contrib = pd.Series({"STRATEGY": 0.1, "STRATEGY_SEGMENT": 0.3, "PRODUCT": -0.2, "CHANNEL": 0.05, "AGE_TAG": -0.01,
                         "SCORE_TAG": -0.4, "FAMILY": 0.0, "BOOKING_MONTH": 0.02})
    values = pd.Series({v: f"valor_{v}" for v in VARIABLES})
    f = top_factors(contrib)
    assert f["suben"] == ["STRATEGY_SEGMENT", "STRATEGY", "CHANNEL"]
    assert f["bajan"] == ["SCORE_TAG", "PRODUCT", "AGE_TAG"]
    p = reading_phrases(contrib, values)
    assert len(p["suben"]) == 3 and len(p["bajan"]) == 3 and p["nota"] == NOTE
    text = as_text(p)
    assert causal_terms(text) == [] and NOTE in text


def test_detector_de_lenguaje_causal():
    assert causal_terms("El score provoca la mora") == ["provoca"]
    assert causal_terms("Esto se debe a que... debido a X") == ["debido a"]
    assert causal_terms("el modelo lo asocia con una tasa más alta") == []
    assert "causa" in CAUSAL_WORDS


def test_frases_cuando_no_hay_factores_que_bajen():
    contrib = pd.Series({v: 0.1 for v in VARIABLES})
    p = reading_phrases(contrib, pd.Series({v: "x" for v in VARIABLES}))
    assert p["bajan"] == [] and "(ninguno)" in as_text(p)
