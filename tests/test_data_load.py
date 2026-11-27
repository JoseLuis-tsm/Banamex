"""Pruebas del cargador (src/data/load.py): alias dummiN, nombres reales y espacios."""

import pandas as pd
import pytest

from src.data.catalog import ALL_COLUMNS
from src.data.dictionary import column_map, read_dictionary
from src.data.load import FORMAT_ALIAS, FORMAT_REAL, ColumnFormatError, load_raw, load_standard, standardize_columns
from src.utils import get_path, load_config

ALIAS_TO_REAL = column_map(read_dictionary(get_path(load_config(), "dictionary")))
REAL_TO_ALIAS = {v: k for k, v in ALIAS_TO_REAL.items()}


def _fila():
    """Una fila válida con todos los campos del contrato (nombres reales)."""
    fila = {c: 0 for c in ALL_COLUMNS}
    fila.update({
        "VINTAGE": 202401, "STRATEGY": "14.Recalibration 1.0", "STRATEGY_SEGMENT": "Brands-NonPayroll-OffUs&LE ",
        "PRODUCT": "Clasica", "CHANNEL": "Branch", "AGE_TAG": "01| Young < 25", "SCORE_TAG": "151-160",
        "ACCOUNTS": 5, "CREDITLINESAMTS": 100000, "ENRACCTS": 2,
    })
    return fila


def test_nombres_reales_se_reconocen():
    renames, info = standardize_columns(ALL_COLUMNS, ALIAS_TO_REAL)
    assert info["formato"] == FORMAT_REAL
    assert set(renames.values()) == set(ALL_COLUMNS)
    assert info["faltantes"] == [] and info["no_declaradas"] == []


def test_alias_dummi_se_renombran():
    header = [REAL_TO_ALIAS.get(c, c) for c in ALL_COLUMNS]  # MOB se queda como MOB
    renames, info = standardize_columns(header + ["extra"], ALIAS_TO_REAL)
    assert info["formato"] == FORMAT_ALIAS
    assert renames["dummi1"] == "VINTAGE" and renames["MOB"] == "MOB"
    assert info["no_declaradas"] == ["extra"]


def test_mezcla_de_formatos_falla():
    header = ["VINTAGE", "dummi12"] + ALL_COLUMNS[2:]
    with pytest.raises(ColumnFormatError, match="mezcla"):
        standardize_columns(header, ALIAS_TO_REAL)


def test_sin_mob_falla_con_mensaje_claro():
    with pytest.raises(ColumnFormatError, match="MOB"):
        standardize_columns([c for c in ALL_COLUMNS if c != "MOB"], ALIAS_TO_REAL)


def test_columna_faltante_se_informa():
    _, info = standardize_columns([c for c in ALL_COLUMNS if c != "REVENUE"], ALIAS_TO_REAL)
    assert info["faltantes"] == ["REVENUE"]


@pytest.mark.parametrize("con_alias", [False, True])
def test_load_raw_renombra_y_quita_espacios(tmp_path, con_alias):
    df = pd.DataFrame([_fila()])
    if con_alias:
        df = df.rename(columns=REAL_TO_ALIAS)
    df["no_declarada"] = 1
    path = tmp_path / "datos.csv"
    df.to_csv(path, index=False, encoding="utf-8")

    cargado, info = load_raw(path, ALIAS_TO_REAL)
    assert list(cargado.columns) == ALL_COLUMNS
    assert cargado.loc[0, "STRATEGY_SEGMENT"] == "Brands-NonPayroll-OffUs&LE"
    assert info["filas_con_espacios"]["STRATEGY_SEGMENT"] == 1
    assert info["no_declaradas"] == ["no_declarada"]
    assert str(cargado["PRODUCT"].dtype) == "category"

    # Ida y vuelta por la base estándar
    std = tmp_path / "std.csv"
    cargado.to_csv(std, index=False, encoding="utf-8")
    assert load_standard(std).loc[0, "SCORE_TAG"] == "151-160"
