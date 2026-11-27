"""Pruebas de la explicación de S1 (src/explain/s1_factors.py): descomposición exacta y frases."""

import numpy as np
import pandas as pd
import pytest

from src.explain import s1_factors
from src.explain.phrases import causal_terms
from src.explain.shap_values import NOTE, VARIABLES, variable_values
from src.models.baselines import RatioBaseline


def _p(product, score, **kw):
    base = {"VINTAGE": 202601, "STRATEGY": "r.Reca champ", "STRATEGY_SEGMENT": "Costco", "PRODUCT": product, "CHANNEL": "Costco",
            "AGE_TAG": "01| Young < 25", "SCORE_TAG": score}
    return {**base, **kw}


@pytest.fixture
def model():
    train = pd.DataFrame([
        _p("Oro", "<130", ACCOUNTS=10, CREDITLINESAMTS=1_000),       # r_ps = 100
        _p("Oro", "130-140", ACCOUNTS=10, CREDITLINESAMTS=3_000),    # r_ps = 300; r_Oro = 200
        _p("Clasica", "<130", ACCOUNTS=20, CREDITLINESAMTS=2_000),   # r_ps = r_Clasica = 100
    ])  # global = 6000 / 40 = 150
    return RatioBaseline([["PRODUCT", "SCORE_TAG"], ["PRODUCT"], []], "CREDITLINESAMTS", "ACCOUNTS").fit(train)


def test_descomposicion_exacta_con_los_tres_niveles(model):
    profiles = pd.DataFrame([_p("Oro", "130-140"), _p("Oro", "151-160"), _p("Joy", "<130")])
    d = s1_factors.decompose(model, profiles)
    assert list(d["NIVEL"]) == [0, 1, 2]
    assert np.exp(d["VALOR_BASE"]).tolist() == pytest.approx([150] * 3)
    assert np.exp(d["PRODUCT"]).tolist() == pytest.approx([200 / 150, 200 / 150, 1.0])
    assert np.exp(d["SCORE_TAG"]).tolist() == pytest.approx([300 / 200, 1.0, 1.0])
    assert np.exp(d[s1_factors.SALIDA_LOG]).tolist() == pytest.approx([300, 200, 150])
    assert s1_factors.additivity_error(d) < 1e-12
    others = [v for v in VARIABLES if v not in s1_factors.S1_VARIABLES]
    assert (d[others] == 0).all(axis=None)


def test_rechaza_otra_estructura_de_niveles():
    other = RatioBaseline([["PRODUCT"], []], "CREDITLINESAMTS", "ACCOUNTS")
    other.tables = [(["PRODUCT"], {"Oro": 1.0}), ([], {(): 1.0})]
    with pytest.raises(ValueError):
        s1_factors.decompose(other, pd.DataFrame([_p("Oro", "<130")]))


def test_frases_de_s1_con_factores_y_sin_lenguaje_causal(model):
    profiles = pd.DataFrame([_p("Oro", "<130"), _p("Joy", "<130")])
    d = s1_factors.decompose(model, profiles)
    values = variable_values(profiles)
    oro = s1_factors.reading_phrases(d.loc[0, VARIABLES], values.loc[0])
    assert "1.33 veces el promedio general" in oro["suben"][0]
    assert "dentro de Oro" in oro["bajan"][0] and "0.50 veces la del producto" in oro["bajan"][0]
    assert oro["nota"] == NOTE and not causal_terms(s1_factors.as_text(oro))
    joy = s1_factors.reading_phrases(d.loc[1, VARIABLES], values.loc[1])
    assert joy["suben"] == [] and joy["bajan"] == []  # producto no visto: todo es el promedio general
