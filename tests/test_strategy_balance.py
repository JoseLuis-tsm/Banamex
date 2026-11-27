"""Pruebas del diagnóstico de balance (src/strategy/balance.py)."""

import math

import pandas as pd
import pytest

from src.strategy.balance import (
    categorical_balance, continuous_balance, smd_continuous, smd_proportion, weighted_mean_sd,
)


def test_smd_de_proporciones():
    assert smd_proportion(0.5, 0.5) == 0.0
    assert smd_proportion(0.0, 0.0) == 0.0
    assert smd_proportion(0.6, 0.4) == pytest.approx(0.2 / math.sqrt(0.24))
    assert smd_proportion(0.4, 0.6) == pytest.approx(-smd_proportion(0.6, 0.4))


def test_media_y_sd_ponderadas_y_smd_continua():
    mean, sd = weighted_mean_sd(pd.Series([0.0, 1.0]), pd.Series([1, 3]))
    assert mean == pytest.approx(0.75) and sd == pytest.approx(math.sqrt(0.1875))
    assert smd_continuous(1.0, 1.0, 0.0, 1.0) == pytest.approx(1.0)


def test_balance_categorico_ponderado_por_cuentas():
    df = pd.DataFrame({
        "S": ["a", "a", "b", "b"], "X": ["p", "q", "p", "q"], "ACCOUNTS": [1, 3, 1, 3],
    })
    t = categorical_balance(df, "X", "S", "a", "b").set_index("categoria")
    assert t.loc["q", "participacion_a"] == pytest.approx(0.75)
    assert (t["smd"] == 0).all() and not t["desbalance"].any()

    df.loc[2, "ACCOUNTS"] = 3  # b: p=3/6
    t = categorical_balance(df, "X", "S", "a", "b").set_index("categoria")
    assert t.loc["p", "diferencia_pp"] == pytest.approx(100 * (0.25 - 0.5))
    assert t.loc["p", "desbalance"]


def test_balance_categorico_con_categoria_ausente_en_un_grupo():
    df = pd.DataFrame({"S": ["a", "a", "b"], "X": ["p", "q", "p"], "ACCOUNTS": [1, 1, 2]})
    t = categorical_balance(df, "X", "S", "a", "b").set_index("categoria")
    assert t.loc["q", "participacion_b"] == 0


def test_balance_continuo():
    df = pd.DataFrame({"S": ["a", "b"], "V": [1.0, 1.0], "ACCOUNTS": [1, 1]})
    assert continuous_balance(df, "V", "S", "a", "b")["smd"].iloc[0] == 0.0
