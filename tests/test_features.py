"""Pruebas de variables de entrada y partición (src/features/engineering.py)."""

import pandas as pd
import pytest

from src.data import catalog as cat
from src.features.engineering import (
    AGE_FEATURE, FAMILY_FEATURE, FORBIDDEN_INPUTS, MONTH_FEATURE, SCORE_FEATURE, TEST, TRAIN, VALIDATION,
    assign_split, build_features, feature_columns, leakage_problems, model_frame, one_hot_name, score_feature_index,
)
from src.utils import load_config

SPLIT = load_config()["split"]


def _perfiles():
    return pd.DataFrame([
        {"VINTAGE": 202401, "STRATEGY": "14.Recalibration 1.0", "STRATEGY_SEGMENT": "Costco", "PRODUCT": "Costco",
         "CHANNEL": "Costco", "AGE_TAG": "01| Young < 25", "SCORE_TAG": "<130"},
        {"VINTAGE": 202512, "STRATEGY": "DARIA 1.6", "STRATEGY_SEGMENT": "LE Thin", "PRODUCT": "Home Depot",
         "CHANNEL": "Home Depot", "AGE_TAG": "03| Mature >30", "SCORE_TAG": "i.205+"},
    ])


def test_las_variables_son_unicas_y_ninguna_esta_prohibida():
    cols = feature_columns()
    assert len(cols) == len(set(cols))
    assert not set(cols) & set(FORBIDDEN_INPUTS)
    assert cols[score_feature_index()] == SCORE_FEATURE


def test_build_features_codifica_cada_variable():
    x = build_features(_perfiles())
    assert list(x.columns) == feature_columns()
    assert x.loc[0, one_hot_name("PRODUCT", "Costco")] == 1 and x.loc[1, one_hot_name("PRODUCT", "Home Depot")] == 1
    assert x.loc[1, one_hot_name("STRATEGY_SEGMENT", "LE Thin")] == 1
    assert list(x[AGE_FEATURE]) == [1, 3]
    assert list(x[SCORE_FEATURE]) == [0.0, 1.0]
    assert list(x[FAMILY_FEATURE]) == [0, 1]
    assert list(x[MONTH_FEATURE]) == [1, 12]
    # Una sola categoría activa por grupo de one-hot
    for col in ("STRATEGY", "STRATEGY_SEGMENT", "PRODUCT", "CHANNEL"):
        assert (x[[c for c in x.columns if c.startswith(col + "__")]].sum(axis=1) == 1).all()


def test_build_features_rechaza_valores_fuera_de_catalogo():
    df = _perfiles()
    df.loc[0, "PRODUCT"] = "Inventado"
    with pytest.raises(ValueError):
        build_features(df)


def test_particion_temporal_coincide_con_la_config():
    v = pd.Series([202401, 202508, 202509, 202511, 202512, 202603, 202604, 202609])
    assert list(assign_split(v, SPLIT, "test")) == [TRAIN, TRAIN, VALIDATION, VALIDATION, TEST, TEST, None, None] or \
        list(assign_split(v, SPLIT, "test").fillna("-")) == [TRAIN, TRAIN, VALIDATION, VALIDATION, TEST, TEST, "-", "-"]
    assert list(assign_split(v, SPLIT, "test_s1"))[-2:] == [TEST, TEST]


def test_la_prueba_de_fuga_detecta_problemas():
    df = _perfiles()
    x = build_features(df)
    splits = pd.Series([TRAIN, TEST])
    assert leakage_problems(x, splits, df["VINTAGE"]) == []
    assert leakage_problems(x.assign(MOB=0), splits, df["VINTAGE"])  # columna prohibida
    assert leakage_problems(x, pd.Series([TEST, TRAIN]), df["VINTAGE"])  # orden temporal invertido
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert leakage_problems(build_features(dup), pd.Series([TRAIN, TEST, TEST]), dup["VINTAGE"])  # vintage repetida


def test_model_frame_descarta_vintages_fuera_de_conjuntos():
    df = _perfiles()
    df.loc[1, "VINTAGE"] = 202605  # no cae en ningún conjunto de S3
    frame, problems = model_frame(df, SPLIT, "test")
    assert problems == [] and len(frame) == 1


def test_todas_las_categorias_del_catalogo_tienen_columna():
    cols = set(feature_columns())
    for v in cat.PRODUCTS:
        assert one_hot_name("PRODUCT", v) in cols
    for fam in cat.SEGMENTS_BY_FAMILY.values():
        for v in fam:
            assert one_hot_name("STRATEGY_SEGMENT", v) in cols
