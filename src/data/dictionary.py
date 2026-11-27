"""Lectura del diccionario de variables y extracción de sus catálogos.

El diccionario (data/contracts/diccionario_variables.csv) trae por variable un alias
`dummiN`, el nombre real, una descripción con los valores permitidos entre comillas y
los valores mínimo y máximo. Aquí se lee tal cual; catalog.py guarda la versión limpia.
"""

import re
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd


def read_dictionary(path: Union[str, Path]) -> pd.DataFrame:
    """Lee el diccionario en UTF-8 con columnas: alias, variable, descripcion, minimo, maximo.

    El alias queda vacío ("") para las variables que no lo traen (MOB).
    """
    raw = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    raw.columns = ["alias", "variable", "descripcion", "minimo", "maximo"]
    for col in ["alias", "variable"]:
        raw[col] = raw[col].str.strip()
    return raw


def column_map(dictionary: pd.DataFrame) -> Dict[str, str]:
    """Mapa alias → nombre real, solo para las variables que tienen alias."""
    with_alias = dictionary[dictionary["alias"] != ""]
    return dict(zip(with_alias["alias"], with_alias["variable"]))


def _quoted(text: str) -> List[str]:
    """Valores entre comillas dobles de un texto, en el orden en que aparecen."""
    return re.findall(r'"([^"]*)"', text)


def _description(dictionary: pd.DataFrame, variable: str) -> str:
    return dictionary.loc[dictionary["variable"] == variable, "descripcion"].iloc[0]


def _values_by_family(description: str) -> Dict[str, List[str]]:
    """Separa una descripción con bloques '- Si STRATEGY es ... puede tomar un valor de entre: (...)'.

    Devuelve {'Recalibration': [...], 'DARIA': [...]} con los valores de cada bloque
    (los que vienen después de los dos puntos, no los nombres de estrategia de la condición).
    """
    out: Dict[str, List[str]] = {}
    for block in description.split("- Si")[1:]:
        condition, _, values = block.partition(":")
        family = "DARIA" if "DARIA" in condition else "Recalibration"
        out[family] = _quoted(values)
    return out


def dictionary_catalog(dictionary: pd.DataFrame) -> Dict[str, object]:
    """Catálogos tal como vienen en el diccionario, **sin** quitar espacios.

    Devuelve un diccionario con: products, channels, age_tags, segments_by_family,
    scores_by_family, strategy_windows (lista de (inicio, fin, estrategias)) y ranges
    ({variable: (minimo, maximo)} para las que tienen rango numérico).
    """
    windows = []
    for line in _description(dictionary, "STRATEGY").split("\n"):
        match = re.search(r"entre (\d{6}) y (\d{6})", line)
        if match:
            # Las estrategias de la línea son los textos entre comillas.
            windows.append((int(match.group(1)), int(match.group(2)), tuple(_quoted(line))))

    ranges = {}
    for _, row in dictionary.iterrows():
        if row["minimo"] not in ("", "NA") and row["maximo"] not in ("", "NA"):
            ranges[row["variable"]] = (float(row["minimo"]), float(row["maximo"]))

    return {
        "products": _quoted(_description(dictionary, "PRODUCT")),
        "channels": _quoted(_description(dictionary, "CHANNEL")),
        "age_tags": _quoted(_description(dictionary, "AGE_TAG")),
        "segments_by_family": _values_by_family(_description(dictionary, "STRATEGY_SEGMENT")),
        "scores_by_family": _values_by_family(_description(dictionary, "SCORE_TAG")),
        "strategy_windows": windows,
        "ranges": ranges,
    }


def values_with_trailing_spaces(catalog: Dict[str, object]) -> List[str]:
    """Valores de catálogo del diccionario que traen espacios al inicio o al final."""
    values: List[str] = list(catalog["products"]) + list(catalog["channels"]) + list(catalog["age_tags"])
    for key in ("segments_by_family", "scores_by_family"):
        for vals in catalog[key].values():
            values += vals
    return [v for v in values if v != v.strip()]


def catalog_differences(dict_catalog: Dict[str, object]) -> List[str]:
    """Compara los catálogos del diccionario (con strip) contra src/data/catalog.py.

    Devuelve una lista de diferencias legibles; vacía si coinciden.
    """
    from src.data import catalog as cat  # import local: catalog no depende de este módulo

    def strip_all(values):
        return [v.strip() for v in values]

    diffs: List[str] = []
    pairs = [
        ("PRODUCT", strip_all(dict_catalog["products"]), cat.PRODUCTS),
        ("CHANNEL", strip_all(dict_catalog["channels"]), cat.CHANNELS),
        ("AGE_TAG", strip_all(dict_catalog["age_tags"]), cat.AGE_TAGS),
    ]
    for family in (cat.RECALIBRATION, cat.DARIA):
        pairs.append((f"STRATEGY_SEGMENT/{family}", strip_all(dict_catalog["segments_by_family"].get(family, [])), cat.SEGMENTS_BY_FAMILY[family]))
        pairs.append((f"SCORE_TAG/{family}", strip_all(dict_catalog["scores_by_family"].get(family, [])), cat.SCORES_BY_FAMILY[family]))
    for name, from_dict, from_cat in pairs:
        if from_dict != from_cat:
            diffs.append(f"{name}: diccionario {from_dict} ≠ catalog.py {from_cat}")

    windows = [(s, e, tuple(strats)) for s, e, strats, _ in cat.STRATEGY_WINDOWS]
    if dict_catalog["strategy_windows"] != windows:
        diffs.append(f"STRATEGY por vintage: diccionario {dict_catalog['strategy_windows']} ≠ catalog.py {windows}")

    expected_ranges = {"VINTAGE": (cat.VINTAGE_MIN, cat.VINTAGE_MAX), "MOB": (cat.MOB_MIN, cat.MOB_MAX)}
    for col in cat.VOLUME_COLUMNS + cat.RESULT_COLUMNS:
        expected_ranges[col] = (cat.AMOUNT_MIN, cat.AMOUNT_MAX)
    for col, rng in expected_ranges.items():
        if dict_catalog["ranges"].get(col) != (float(rng[0]), float(rng[1])):
            diffs.append(f"Rango de {col}: diccionario {dict_catalog['ranges'].get(col)} ≠ catalog.py {rng}")
    return diffs
