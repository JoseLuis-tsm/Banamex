"""Madurez de cohortes, curvas de Over30 y conjuntos de modelado de S1 y S3.

Una cohorte es madura a horizonte H si ya se observó hasta MOB = H, es decir, si el
MOB máximo de su vintage (meses entre la vintage y el corte) es al menos H.
"""

from typing import Iterable, List

import numpy as np
import pandas as pd

from src.data import catalog as cat

# Columnas de los conjuntos de modelado (además de las 7 dimensiones de la cohorte).
S1_TARGET = "LINEA_PROMEDIO"
S3_TARGET = "TASA_OVER30"


def is_mature(vintage: int, horizon: int, cutoff: int = cat.VINTAGE_MAX) -> bool:
    """True si la vintage ya se observó hasta MOB = horizon con el corte dado."""
    return cat.max_mob(vintage, cutoff) >= horizon


def mature_vintages(vintages: Iterable[int], horizon: int, cutoff: int = cat.VINTAGE_MAX) -> List[int]:
    """Vintages maduras a `horizon`, ordenadas y sin repetir."""
    return sorted({int(v) for v in vintages if is_mature(v, horizon, cutoff)})


def maturity_table(df: pd.DataFrame, horizon: int, cutoff: int = cat.VINTAGE_MAX) -> pd.DataFrame:
    """Por vintage: MOB máximo posible y observado, si es madura y qué estrategias tiene."""
    t = df.groupby("VINTAGE").agg(mob_max_observado=("MOB", "max")).reset_index()
    t["mob_max_posible"] = t["VINTAGE"].map(lambda v: cat.max_mob(v, cutoff))
    t["madura"] = t["mob_max_posible"] >= horizon
    strategies = df.groupby("VINTAGE")["STRATEGY"].agg(lambda s: " / ".join(sorted(set(s.astype(str)))))
    t["estrategias"] = t["VINTAGE"].map(strategies)
    t["tipo_asignacion"] = t["VINTAGE"].map(cat.assignment_type)
    return t


def build_s1_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Conjunto de S1: una fila por cohorte en MOB = 0 con la línea promedio asignada.

    Fórmula: LINEA_PROMEDIO = CREDITLINESAMTS / ACCOUNTS (en MOB = 0, la línea del booking).
    Incluye todas las vintages (también DARIA 2.0: tiene línea aunque no tenga riesgo).
    ACCOUNTS se conserva como peso.
    """
    booked = df.loc[df["MOB"] == 0, cat.COHORT_COLUMNS + ["ACCOUNTS", "CREDITLINESAMTS"]].copy()
    booked[S1_TARGET] = booked["CREDITLINESAMTS"] / booked["ACCOUNTS"]
    return booked.sort_values(cat.COHORT_COLUMNS).reset_index(drop=True)


def build_s3_dataset(df: pd.DataFrame, horizon: int, cutoff: int = cat.VINTAGE_MAX) -> pd.DataFrame:
    """Conjunto de S3: una fila por cohorte madura en MOB = horizon con su tasa de Over30.

    Fórmula: TASA_OVER30 = OVER30ACCTS / ENRACCTS en MOB = H, donde OVER30ACCTS es el
    acumulado de cuentas que llegaron a Over30 hasta H y ENRACCTS las cuentas con saldo en H.
    Excluye las estrategias no evaluables en riesgo (DARIA 2.0) y las filas con
    ENRACCTS = 0 (tasa indefinida). ENRACCTS se conserva como peso.
    """
    keep = (
        (df["MOB"] == horizon)
        & df["VINTAGE"].map(lambda v: is_mature(v, horizon, cutoff))
        & ~df["STRATEGY"].astype(str).isin(cat.NOT_RISK_EVALUABLE)
        & (df["ENRACCTS"] > 0)
    )
    out = df.loc[keep, cat.COHORT_COLUMNS + ["ACCOUNTS", "ENRACCTS", "OVER30ACCTS"]].copy()
    out[S3_TARGET] = out["OVER30ACCTS"] / out["ENRACCTS"]
    return out.sort_values(cat.COHORT_COLUMNS).reset_index(drop=True)


def over30_curves(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Tasa acumulada de Over30 por MOB para cada grupo `by`.

    Fórmulas (sumas dentro del grupo y MOB):
        tasa_over30_enr = Σ OVER30ACCTS / Σ ENRACCTS   (definición de S3)
        tasa_over30_cuentas = Σ OVER30ACCTS / Σ ACCOUNTS (cuentas colocadas, referencia)
    """
    g = df.groupby(by + ["MOB"], observed=True)[["OVER30ACCTS", "ENRACCTS", "ACCOUNTS"]].sum().reset_index()
    g["tasa_over30_enr"] = g["OVER30ACCTS"] / g["ENRACCTS"].replace(0, np.nan)
    g["tasa_over30_cuentas"] = g["OVER30ACCTS"] / g["ACCOUNTS"]
    return g


def over30_exceeds_enr(df: pd.DataFrame, mob: int, by: List[str] = None) -> pd.DataFrame:
    """Proporción de cohortes con OVER30ACCTS > ENRACCTS en un MOB (total o por grupo).

    Devuelve columnas: [by...], cohortes, cohortes_over30_mayor_enr, proporcion.
    """
    rows = df[df["MOB"] == mob].assign(excede=lambda d: d["OVER30ACCTS"] > d["ENRACCTS"])
    if by:
        t = rows.groupby(by, observed=True)["excede"].agg(["size", "sum"]).reset_index()
    else:
        t = pd.DataFrame({"size": [len(rows)], "sum": [int(rows["excede"].sum())]})
    t = t.rename(columns={"size": "cohortes", "sum": "cohortes_over30_mayor_enr"})
    t["proporcion"] = t["cohortes_over30_mayor_enr"] / t["cohortes"].replace(0, np.nan)
    return t
