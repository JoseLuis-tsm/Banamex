"""Carga del CSV de datos con nombres de columna estándar (los nombres reales).

El CSV puede traer los nombres reales (VINTAGE, STRATEGY, ...) o los alias `dummiN`
del diccionario. Ambos se aceptan: se renombra a los nombres reales y se informa cuál
se encontró. MOB no tiene alias: se busca una columna llamada exactamente "MOB".
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from src.data.catalog import ALL_COLUMNS, TEXT_DIMENSIONS

FORMAT_REAL = "nombres reales"
FORMAT_ALIAS = "alias dummiN"


class ColumnFormatError(ValueError):
    """El encabezado no permite identificar las columnas del contrato."""


def standardize_columns(columns: List[str], alias_to_real: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, object]]:
    """Decide cómo renombrar un encabezado a los nombres reales.

    Parámetros:
        columns: encabezado del archivo (se comparan sin espacios a los lados).
        alias_to_real: mapa alias → nombre real (de column_map.json o del diccionario).

    Devuelve:
        (renombres, info). `renombres` es {columna del archivo: nombre real} solo para las
        columnas reconocidas. `info` trae: formato (nombres reales o alias dummiN),
        no_declaradas (columnas que no están en el diccionario; se descartan) y
        faltantes (columnas del contrato que no aparecen).

    Lanza ColumnFormatError si el archivo mezcla alias y nombres reales, si no reconoce
    ninguna columna o si no hay columna MOB.
    """
    real_names = set(ALL_COLUMNS)
    renames: Dict[str, str] = {}
    n_real = n_alias = 0
    undeclared: List[str] = []
    for col in columns:
        clean = col.strip()
        if clean in real_names:
            renames[col] = clean
            n_real += clean != "MOB"  # MOB no tiene alias: no cuenta para decidir el formato
        elif clean in alias_to_real:
            renames[col] = alias_to_real[clean]
            n_alias += 1
        else:
            undeclared.append(col)

    if n_real and n_alias:
        raise ColumnFormatError(
            f"El archivo mezcla nombres reales ({n_real}) y alias dummiN ({n_alias}). "
            "Usa un solo formato en el encabezado."
        )
    if not n_real and not n_alias:
        raise ColumnFormatError("No se reconoció ninguna columna del diccionario en el encabezado.")
    if "MOB" not in renames.values():
        raise ColumnFormatError(
            "No hay una columna llamada 'MOB' (el diccionario no le da alias). "
            "Revisa el encabezado e indica cuál es la columna del mes de desempeño."
        )

    found = set(renames.values())
    info = {
        "formato": FORMAT_REAL if n_real else FORMAT_ALIAS,
        "no_declaradas": undeclared,
        "faltantes": [c for c in ALL_COLUMNS if c not in found],
    }
    return renames, info


def strip_text_dimensions(df: pd.DataFrame) -> Dict[str, int]:
    """Quita espacios a los lados de las dimensiones de texto (en el mismo DataFrame).

    Devuelve {columna: filas que cambiaron} para documentar la normalización.
    """
    changed = {}
    for col in TEXT_DIMENSIONS:
        if col in df.columns:
            original = df[col].astype("object")
            stripped = original.str.strip()
            changed[col] = int(((original != stripped) & original.notna()).sum())
            df[col] = stripped
    return changed


def load_raw(path: Union[str, Path], alias_to_real: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Carga el CSV crudo en UTF-8, renombra a nombres reales y normaliza espacios.

    Las dimensiones de texto se leen como texto, se limpian y quedan como `category`
    (ahorra memoria: hay ~1 millón de filas con pocos valores distintos).

    Devuelve:
        (df, info). `df` solo tiene las columnas del contrato que aparecieron;
        `info` agrega a lo de standardize_columns() las filas cambiadas por strip.
    """
    header = pd.read_csv(path, encoding="utf-8", nrows=0).columns.tolist()
    renames, info = standardize_columns(header, alias_to_real)
    text_source_cols = [src for src, real in renames.items() if real in TEXT_DIMENSIONS]
    df = pd.read_csv(
        path, encoding="utf-8", usecols=list(renames), dtype={c: str for c in text_source_cols}
    )
    df = df.rename(columns=renames)
    df = df[[c for c in ALL_COLUMNS if c in df.columns]]
    info["filas_con_espacios"] = strip_text_dimensions(df)
    for col in TEXT_DIMENSIONS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df, info


def load_standard(path: Union[str, Path]) -> pd.DataFrame:
    """Carga la base estándar (data/processed/base_estandar.csv) con tipos económicos.

    Las dimensiones de texto quedan como `category`; las numéricas, con el tipo que
    infiere pandas (int64 para conteos, float64 para montos).
    """
    dtypes = {c: "category" for c in TEXT_DIMENSIONS}
    df = pd.read_csv(path, encoding="utf-8", dtype=dtypes)
    return df[[c for c in ALL_COLUMNS if c in df.columns] + [c for c in df.columns if c not in ALL_COLUMNS]]

