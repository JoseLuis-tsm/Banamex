"""Variables de entrada de los modelos y partición temporal por vintage (CLAUDE.md §7).

Entran: STRATEGY, STRATEGY_SEGMENT, PRODUCT y CHANNEL en one-hot; AGE_TAG como ordinal
(1 a 3); el score como posición relativa 0–1 dentro de su familia, más la familia; y el
mes de booking (1–12) derivado de la vintage.

No entran: la vintage como número, el MOB, ACCOUNTS y ENRACCTS (son pesos) ni ninguna
columna de resultado.

El one-hot usa las categorías **del catálogo** (src/data/catalog.py), no las que aparezcan
en los datos: así no se "ajusta" nada con los datos (no hay fuga posible) y las columnas
son siempre las mismas, también en la API.
"""

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data import catalog as cat

ONE_HOT_SOURCES: Dict[str, List[str]] = {
    "STRATEGY": cat.STRATEGIES,
    "STRATEGY_SEGMENT": cat.SEGMENTS_BY_FAMILY[cat.RECALIBRATION] + cat.SEGMENTS_BY_FAMILY[cat.DARIA],
    "PRODUCT": cat.PRODUCTS,
    "CHANNEL": cat.CHANNELS,
}
# El segmento "Affinity" (DARIA) no choca con otro segmento: los de Recalibration son distintos.
AGE_FEATURE = "EDAD_ORDINAL"
SCORE_FEATURE = "SCORE_POSICION"
FAMILY_FEATURE = "FAMILIA_DARIA"
MONTH_FEATURE = "MES_BOOKING"

# Columnas que jamás pueden ser variables de entrada.
FORBIDDEN_INPUTS: List[str] = ["VINTAGE", "MOB"] + cat.VOLUME_COLUMNS + cat.RESULT_COLUMNS + ["LINEA_PROMEDIO", "TASA_OVER30"]

SPLIT_COLUMN = "CONJUNTO"
TRAIN, VALIDATION, TEST = "entrenamiento", "validacion", "prueba"


def _slug(text: str) -> str:
    """Convierte un valor de catálogo en un sufijo de nombre de columna seguro."""
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")


def one_hot_name(column: str, value: str) -> str:
    """Nombre de la columna one-hot, por ejemplo PRODUCT=Home Depot → 'PRODUCT__Home_Depot'."""
    return f"{column}__{_slug(value)}"


def feature_columns() -> List[str]:
    """Lista fija y ordenada de variables de entrada."""
    cols = [one_hot_name(c, v) for c, values in ONE_HOT_SOURCES.items() for v in values]
    return cols + [AGE_FEATURE, SCORE_FEATURE, FAMILY_FEATURE, MONTH_FEATURE]


def score_feature_index() -> int:
    """Posición de la variable de score en feature_columns() (para la restricción monótona)."""
    return feature_columns().index(SCORE_FEATURE)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz de variables a partir de las 7 dimensiones de la cohorte.

    Parámetros:
        df: DataFrame con VINTAGE, STRATEGY, STRATEGY_SEGMENT, PRODUCT, CHANNEL, AGE_TAG y
            SCORE_TAG (nombres reales, sin espacios). Lanza ValueError si algún valor no
            está en el catálogo.

    Devuelve:
        DataFrame float con columnas feature_columns(), mismo índice que df.
    """
    out = pd.DataFrame(index=df.index)
    for column, values in ONE_HOT_SOURCES.items():
        observed = df[column].astype(str)
        unknown = sorted(set(observed) - set(values))
        if unknown:
            raise ValueError(f"{column} tiene valores fuera de catálogo: {unknown[:5]}")
        for value in values:
            out[one_hot_name(column, value)] = (observed == value).astype(float)

    strategy = df["STRATEGY"].astype(str)
    family = strategy.map(cat.STRATEGY_FAMILY)
    # Edad ordinal: 1 = menor de 25, 2 = 25–30, 3 = mayor de 30.
    out[AGE_FEATURE] = df["AGE_TAG"].astype(str).map(cat.age_ordinal).astype(float)
    # Posición del score: i / (n − 1) dentro de la familia (0 = score más bajo).
    out[SCORE_FEATURE] = [cat.score_position(s, f) for s, f in zip(df["SCORE_TAG"].astype(str), family)]
    out[FAMILY_FEATURE] = (family == cat.DARIA).astype(float)
    # Mes de booking 1–12: estacionalidad, sin dar al modelo la vintage como número.
    out[MONTH_FEATURE] = df["VINTAGE"].astype(int).map(cat.booking_month).astype(float)
    return out[feature_columns()]


def assign_split(vintages: pd.Series, split: Dict[str, Dict[str, int]], test_key: str = "test") -> pd.Series:
    """Asigna cada vintage a entrenamiento, validación o prueba según config["split"].

    Parámetros:
        vintages: Serie de vintages AAAAMM.
        split: config["split"] con rangos {start, end} para train, validation y test/test_s1.
        test_key: "test" para S3 (hasta 202603) o "test_s1" para S1 (hasta 202609).

    Devuelve:
        Serie de texto (entrenamiento, validacion, prueba); NaN si la vintage no cae en ninguno.
    """
    out = pd.Series(np.nan, index=vintages.index, dtype=object)
    for name, key in ((TRAIN, "train"), (VALIDATION, "validation"), (TEST, test_key)):
        rng = split[key]
        out[vintages.between(rng["start"], rng["end"])] = name
    return out


def leakage_problems(features: pd.DataFrame, splits: pd.Series, vintages: pd.Series) -> List[str]:
    """Comprueba que no haya fuga de información. Devuelve la lista de problemas (vacía = ok).

    Reglas: ninguna vintage en dos conjuntos; ninguna columna prohibida (resultado, MOB,
    vintage numérica, pesos) entre las variables; las variables son exactamente
    feature_columns(); los conjuntos respetan el orden temporal (train < validación < prueba).
    """
    problems: List[str] = []
    per_vintage = pd.DataFrame({"v": vintages, "s": splits}).dropna().groupby("v")["s"].nunique()
    shared = per_vintage[per_vintage > 1].index.tolist()
    if shared:
        problems.append(f"Vintages en más de un conjunto: {shared}")
    forbidden = [c for c in features.columns if c in FORBIDDEN_INPUTS]
    if forbidden:
        problems.append(f"Columnas prohibidas entre las variables: {forbidden}")
    if list(features.columns) != feature_columns():
        problems.append("Las variables no coinciden con feature_columns().")
    ranges = pd.DataFrame({"v": vintages, "s": splits}).dropna().groupby("s")["v"].agg(["min", "max"])
    order = [s for s in (TRAIN, VALIDATION, TEST) if s in ranges.index]
    for a, b in zip(order, order[1:]):
        if ranges.loc[a, "max"] >= ranges.loc[b, "min"]:
            problems.append(f"El conjunto {a} no es anterior a {b}.")
    return problems


def model_frame(dataset: pd.DataFrame, split: Dict[str, Dict[str, int]], test_key: str) -> Tuple[pd.DataFrame, List[str]]:
    """Une dimensiones, conjunto y variables en una sola tabla de modelado.

    Devuelve (tabla, problemas_de_fuga). La tabla conserva las columnas originales del
    dataset (dimensiones, pesos y objetivo), agrega CONJUNTO y las variables, y deja fuera
    las filas cuya vintage no cae en ningún conjunto.
    """
    splits = assign_split(dataset["VINTAGE"], split, test_key)
    features = build_features(dataset)
    problems = leakage_problems(features, splits, dataset["VINTAGE"])
    frame = pd.concat([dataset, splits.rename(SPLIT_COLUMN), features], axis=1)
    return frame[frame[SPLIT_COLUMN].notna()].reset_index(drop=True), problems
