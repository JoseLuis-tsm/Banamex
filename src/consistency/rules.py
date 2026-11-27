"""Reglas de consistencia: corren después de los modelos y antes de mostrar un resultado.

Definiciones aprobadas (Fase 3, ver docs/consistencia_fase3.md):
- Perfil fuera del dominio de un modelo: su combinación (familia, segmento, producto, canal)
  no aparece en los datos con que se entrenó ese modelo. La edad y el score pueden variar.
- Perfiles similares (confianza): misma familia, segmento, producto, canal y edad, con
  cualquier score; se suman las cuentas (ACCOUNTS) de las cohortes maduras a 6 meses.
  Alta con 1,000 o más, Media de 200 a 999, Baja con menos de 200.
- Umbrales de riesgo: los del metadata.json de S3 (percentiles de la tasa predicha en
  entrenamiento; desviación documentada respecto de la propuesta).

Las reglas no aprueban ni rechazan: un riesgo Crítico solo genera una alerta de revisión.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.data import catalog as cat
from src.models.metrics import RISK_LEVELS, risk_level
from src.models.s2 import expected_total_amount

DOMAIN_COLUMNS: List[str] = ["FAMILIA", "STRATEGY_SEGMENT", "PRODUCT", "CHANNEL"]
SIMILAR_COLUMNS: List[str] = DOMAIN_COLUMNS + ["AGE_TAG"]
CONFIDENCE_HIGH = 1000
CONFIDENCE_MEDIUM = 200
CONFIDENCE_LEVELS = ["Alta", "Media", "Baja"]

STATUS_OK = "evaluable"
STATUS_NOT_RISK = "no evaluable en riesgo"
STATUS_OUT_S3 = "fuera de dominio en riesgo"
STATUS_OUT_S1 = "fuera de dominio en línea"
CRITICAL = RISK_LEVELS[-1]
ALERT_TEXT = "Revisión recomendada: riesgo Crítico (la herramienta no aprueba ni rechaza)."


def with_family(df: pd.DataFrame) -> pd.DataFrame:
    """Copia con la columna FAMILIA (Recalibration o DARIA) derivada de STRATEGY."""
    out = df.copy()
    out["FAMILIA"] = out["STRATEGY"].map(cat.STRATEGY_FAMILY)
    return out


def domain_keys(training: pd.DataFrame) -> Set[Tuple[str, ...]]:
    """Combinaciones (familia, segmento, producto, canal) presentes en los datos de entrenamiento."""
    return set(map(tuple, with_family(training)[DOMAIN_COLUMNS].drop_duplicates().to_numpy()))


def in_domain(profiles: pd.DataFrame, keys: Set[Tuple[str, ...]]) -> np.ndarray:
    """True si la combinación (familia, segmento, producto, canal) del perfil está en `keys`."""
    combos = with_family(profiles)[DOMAIN_COLUMNS].to_numpy()
    return np.array([tuple(c) in keys for c in combos], dtype=bool)


def similar_accounts_table(mature: pd.DataFrame) -> pd.Series:
    """Cuentas maduras por grupo de perfiles similares.

    Fórmula: para el grupo g = (familia, segmento, producto, canal, edad),
    cuentas_g = Σ ACCOUNTS de las cohortes maduras del grupo (todas las vintages y scores).
    """
    return with_family(mature).groupby(SIMILAR_COLUMNS)["ACCOUNTS"].sum()


def similar_accounts(profiles: pd.DataFrame, table: pd.Series) -> np.ndarray:
    """Cuentas maduras en perfiles similares a cada perfil (0 si no hay ninguno)."""
    idx = pd.MultiIndex.from_frame(with_family(profiles)[SIMILAR_COLUMNS])
    return table.reindex(idx).fillna(0).to_numpy(dtype=float)


def confidence_level(accounts: np.ndarray) -> np.ndarray:
    """Alta (≥ 1,000), Media (200–999) o Baja (< 200)."""
    a = np.asarray(accounts, dtype=float)
    return np.where(a >= CONFIDENCE_HIGH, "Alta", np.where(a >= CONFIDENCE_MEDIUM, "Media", "Baja"))


def apply_rules(
    profiles: pd.DataFrame,
    line: np.ndarray,
    rate: np.ndarray,
    thresholds: Dict[str, float],
    domain_s1: Set[Tuple[str, ...]],
    domain_s3: Set[Tuple[str, ...]],
    similar_table: pd.Series,
    versions: Dict[str, str],
    planned_accounts: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Aplica las reglas de consistencia a un lote de perfiles ya predichos.

    Parámetros:
        profiles: perfiles (las 7 dimensiones de la llave).
        line: S1 por perfil (línea promedio esperada).
        rate: S3 por perfil (tasa calibrada, antes de recortar).
        thresholds: {"p50", "p80", "p95"} del metadata.json de S3.
        domain_s1, domain_s3: domain_keys() de los datos de entrenamiento de cada modelo.
        similar_table: similar_accounts_table() de las cohortes maduras.
        versions: {"modelo_s1", "modelo_s3", "datos_sha256", "horizonte_mob"}.
        planned_accounts: cuentas planeadas (opcional; habilitan S2).

    Devuelve: los perfiles con línea, monto total, tasa recortada a [0, 1], nivel de riesgo,
    alerta, confianza, estado con su motivo y versiones. Lo no evaluable queda vacío (NaN).
    """
    out = profiles.reset_index(drop=True).copy()
    n = len(out)
    risk_ok_strategy = ~out["STRATEGY"].isin(cat.NOT_RISK_EVALUABLE).to_numpy()
    s1_in = in_domain(out, domain_s1)
    s3_in = in_domain(out, domain_s3)
    s3_ok = risk_ok_strategy & s3_in

    # S1 solo dentro de su dominio; S2 = cuentas planeadas × S1 (se calcula, no se predice).
    out["linea_promedio"] = np.where(s1_in, np.asarray(line, dtype=float), np.nan)
    if planned_accounts is not None:
        planned = np.asarray(planned_accounts, dtype=float)
        out["cuentas_planeadas"] = planned
        total = np.full(n, np.nan)
        total[s1_in] = expected_total_amount(planned[s1_in], out.loc[s1_in, "linea_promedio"].to_numpy())
        out["monto_total"] = total
    else:
        out["cuentas_planeadas"] = np.nan
        out["monto_total"] = np.nan

    # S3 recortada a [0, 1] y solo si es evaluable en riesgo y está en el dominio.
    clipped = np.clip(np.asarray(rate, dtype=float), 0.0, 1.0)
    out["tasa_impago"] = np.where(s3_ok, clipped, np.nan)
    levels = np.where(s3_ok, risk_level(np.nan_to_num(clipped), thresholds), None)
    out["nivel_riesgo"] = levels
    out["alerta"] = np.where(levels == CRITICAL, ALERT_TEXT, "")

    accounts = similar_accounts(out, similar_table)
    out["cuentas_similares"] = accounts
    out["confianza"] = confidence_level(accounts)

    status, reason = [], []
    for ok_strategy, in1, in3, strategy in zip(risk_ok_strategy, s1_in, s3_in, out["STRATEGY"]):
        if not in1:
            status.append(STATUS_OUT_S1)
            reason.append("la combinación familia–segmento–producto–canal no aparece en los datos de entrenamiento de S1")
        elif not ok_strategy:
            status.append(STATUS_NOT_RISK)
            reason.append(f"{strategy} solo tiene MOB 0 y 1: sin desempeño de riesgo a 6 meses")
        elif not in3:
            status.append(STATUS_OUT_S3)
            reason.append("la combinación familia–segmento–producto–canal no aparece en los datos de entrenamiento de S3")
        else:
            status.append(STATUS_OK)
            reason.append("")
    out["estado"] = status
    out["motivo"] = reason
    for key, value in versions.items():
        out[key] = value
    return out


def check_rules(out: pd.DataFrame, rate: np.ndarray, line: np.ndarray, thresholds: Dict[str, float]) -> pd.DataFrame:
    """Verifica cada regla fila por fila. Devuelve por regla: filas revisadas y filas que la cumplen."""
    raw_rate = np.asarray(rate, dtype=float)
    ok = out["estado"] == STATUS_OK
    risk = out["tasa_impago"].notna()
    has_line = out["linea_promedio"].notna()
    planned = out["cuentas_planeadas"].notna() & has_line
    expected_level = pd.Series(np.where(risk, risk_level(np.nan_to_num(out["tasa_impago"].to_numpy()), thresholds), None))
    checks = {
        "S2 = cuentas planeadas × S1": (planned, np.isclose(out["monto_total"], out["cuentas_planeadas"] * out["linea_promedio"], rtol=1e-12)),
        "tasa en [0, 1]": (risk, out["tasa_impago"].between(0, 1)),
        "tasa = S3 recortada": (risk, np.isclose(out["tasa_impago"], np.clip(raw_rate, 0, 1))),
        "línea = S1 dentro del dominio": (has_line, np.isclose(out["linea_promedio"], np.asarray(line, dtype=float))),
        "no evaluable en riesgo sin tasa": (out["STRATEGY"].isin(cat.NOT_RISK_EVALUABLE), ~risk),
        "fuera de dominio sin la salida del modelo": (out["estado"].isin([STATUS_OUT_S1, STATUS_OUT_S3]), ~risk),
        "evaluable con tasa y línea": (ok, risk & has_line),
        "nivel de riesgo según umbrales": (risk, out["nivel_riesgo"].eq(expected_level)),
        "alerta si y solo si Crítico": (pd.Series(True, index=out.index), out["alerta"].ne("").eq(out["nivel_riesgo"].eq(CRITICAL))),
        "confianza según cuentas similares": (pd.Series(True, index=out.index), out["confianza"].eq(pd.Series(confidence_level(out["cuentas_similares"])))),
        "estado no evaluable lleva motivo": (~ok, out["motivo"].ne("")),
        "versiones presentes": (pd.Series(True, index=out.index),
                                out[["modelo_s1", "modelo_s3", "datos_sha256", "horizonte_mob"]].notna().all(axis=1)),
    }
    rows = []
    for rule, (mask, passed) in checks.items():
        mask = pd.Series(np.asarray(mask, dtype=bool), index=out.index)
        passed = pd.Series(np.asarray(passed, dtype=bool), index=out.index)
        rows.append({"regla": rule, "filas_revisadas": int(mask.sum()), "filas_que_cumplen": int((mask & passed).sum())})
    t = pd.DataFrame(rows)
    t["cumple_100"] = t["filas_revisadas"] == t["filas_que_cumplen"]
    return t
