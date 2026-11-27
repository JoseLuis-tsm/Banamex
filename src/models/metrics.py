"""Métricas ponderadas de S1 (regresión de la línea) y S3 (tasa de Over30), CLAUDE.md §7.

S1: cada cohorte pesa ACCOUNTS. S3: cada cohorte aporta tantos eventos como cuentas en
Over30 (OVER30ACCTS) y tantos no eventos como cuentas expuestas restantes
(ENRACCTS − OVER30ACCTS), todos con la tasa predicha de la cohorte.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-12


# ===== S1 =====
def regression_metrics(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """MAE, MAPE, R² y sesgo ponderados.

    Fórmulas (w = ACCOUNTS, y = línea observada, ŷ = línea predicha):
        MAE   = Σ w·|ŷ − y| / Σ w
        MAPE  = Σ w·|ŷ − y|/y / Σ w                     (en %, ×100)
        R²    = 1 − Σ w(y − ŷ)² / Σ w(y − ȳ_w)²,  ȳ_w = Σ w·y / Σ w
        sesgo = Σ w(ŷ − y) / Σ w                       (positivo = sobreestima)
        sesgo_pct = Σ w·ŷ / Σ w·y − 1                  (en %, ×100)
    """
    y, pred, w = (np.asarray(a, dtype=float) for a in (y, pred, w))
    err = pred - y
    mean_y = np.sum(w * y) / np.sum(w)
    total_ss = np.sum(w * (y - mean_y) ** 2)
    return {
        "mae": float(np.sum(w * np.abs(err)) / np.sum(w)),
        "mape_pct": float(100 * np.sum(w * np.abs(err) / y) / np.sum(w)),
        "r2": float(1 - np.sum(w * err ** 2) / total_ss) if total_ss > 0 else float("nan"),
        "sesgo": float(np.sum(w * err) / np.sum(w)),
        "sesgo_pct": float(100 * (np.sum(w * pred) / np.sum(w * y) - 1)),
        "peso": float(np.sum(w)),
        "filas": int(len(y)),
    }


# ===== S3 =====
def _expand(events: np.ndarray, exposed: np.ndarray, pred: np.ndarray):
    """Convierte cada cohorte en dos registros: evento (peso = eventos) y no evento (peso = resto)."""
    events, exposed, pred = (np.asarray(a, dtype=float) for a in (events, exposed, pred))
    labels = np.concatenate([np.ones_like(events), np.zeros_like(events)])
    weights = np.concatenate([events, exposed - events])
    scores = np.concatenate([pred, pred])
    keep = weights > 0
    return labels[keep], scores[keep], weights[keep]


def rate_metrics(events: np.ndarray, exposed: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """AUC, KS, Gini, log loss, Brier y razón observado/predicho ponderados.

    Fórmulas (por cohorte i: e_i = OVER30ACCTS, n_i = ENRACCTS, p_i = tasa predicha):
        AUC, KS: sobre registros expandidos (e_i eventos y n_i − e_i no eventos con puntaje p_i);
                 KS = max(TPR − FPR); Gini = 2·AUC − 1
        log loss = −Σ [e_i·log p_i + (n_i − e_i)·log(1 − p_i)] / Σ n_i
        Brier    = Σ [e_i(1 − p_i)² + (n_i − e_i)p_i²] / Σ n_i
        obs/pred = Σ e_i / Σ n_i·p_i
    """
    e, n, p = (np.asarray(a, dtype=float) for a in (events, exposed, pred))
    p = np.clip(p, EPS, 1 - EPS)
    labels, scores, weights = _expand(e, n, p)
    if len(set(labels)) == 2:
        auc = float(roc_auc_score(labels, scores, sample_weight=weights))
        fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
        ks = float(np.max(tpr - fpr))
    else:  # sin eventos o sin no eventos: AUC y KS no están definidos
        auc = ks = float("nan")
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks,
        "log_loss": float(-np.sum(e * np.log(p) + (n - e) * np.log(1 - p)) / np.sum(n)),
        "brier": float(np.sum(e * (1 - p) ** 2 + (n - e) * p ** 2) / np.sum(n)),
        "tasa_observada": float(np.sum(e) / np.sum(n)),
        "tasa_predicha": float(np.sum(n * p) / np.sum(n)),
        "obs_pred": float(np.sum(e) / np.sum(n * p)),
        "peso": float(np.sum(n)),
        "filas": int(len(e)),
    }


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: List[float]) -> np.ndarray:
    """Cuantiles ponderados: el menor valor cuya proporción acumulada de peso alcanza q."""
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="mergesort")
    cum = np.cumsum(weights[order]) / np.sum(weights)
    idx = np.searchsorted(cum, quantiles, side="left")
    return values[order][np.minimum(idx, len(values) - 1)]


def calibration_by_decile(events: np.ndarray, exposed: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Calibración por deciles de la tasa predicha (deciles ponderados por ENRACCTS).

    Por decil: cuentas expuestas, tasa predicha media Σn·p/Σn, tasa observada Σe/Σn y su razón.
    """
    e, n, p = (np.asarray(a, dtype=float) for a in (events, exposed, pred))
    edges = weighted_quantiles(p, n, [i / 10 for i in range(1, 10)])
    decile = np.searchsorted(edges, p, side="left") + 1  # 1 = riesgo predicho más bajo
    df = pd.DataFrame({"decil": decile, "e": e, "n": n, "np": n * p})
    t = df.groupby("decil").agg(cohortes=("e", "size"), cuentas_expuestas=("n", "sum"), eventos=("e", "sum"), esperado=("np", "sum")).reset_index()
    t["tasa_predicha"] = t["esperado"] / t["cuentas_expuestas"]
    t["tasa_observada"] = t["eventos"] / t["cuentas_expuestas"]
    t["obs_pred"] = t["eventos"] / t["esperado"]
    return t.drop(columns="esperado")


RISK_LEVELS = ["Bajo", "Medio", "Alto", "Crítico"]
RISK_PERCENTILES = [0.50, 0.80, 0.95]


def risk_thresholds(rate: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    """Umbrales de riesgo: percentiles 50, 80 y 95 ponderados de la tasa (CLAUDE.md §7)."""
    p50, p80, p95 = weighted_quantiles(rate, weights, RISK_PERCENTILES)
    return {"p50": float(p50), "p80": float(p80), "p95": float(p95)}


def risk_level(rate: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """Nivel de riesgo: Bajo ≤ P50 < Medio ≤ P80 < Alto ≤ P95 < Crítico."""
    edges = [thresholds["p50"], thresholds["p80"], thresholds["p95"]]
    return np.array(RISK_LEVELS)[np.searchsorted(edges, np.asarray(rate, dtype=float), side="left")]


def rate_by_risk_level(events: np.ndarray, exposed: np.ndarray, pred: np.ndarray, thresholds: Dict[str, float]) -> pd.DataFrame:
    """Tasa observada y predicha por nivel de riesgo, y si la observada crece con el nivel."""
    df = pd.DataFrame({"nivel": risk_level(pred, thresholds), "e": events, "n": exposed, "np": np.asarray(exposed) * np.asarray(pred)})
    t = df.groupby("nivel").agg(cohortes=("e", "size"), cuentas_expuestas=("n", "sum"), eventos=("e", "sum"), esperado=("np", "sum"))
    t = t.reindex(RISK_LEVELS).dropna(how="all").reset_index()
    t["tasa_observada"] = t["eventos"] / t["cuentas_expuestas"]
    t["tasa_predicha"] = t["esperado"] / t["cuentas_expuestas"]
    return t.drop(columns="esperado")


def is_increasing(values: List[float]) -> bool:
    """True si la secuencia es estrictamente creciente."""
    return all(b > a for a, b in zip(values, values[1:]))


def score_monotonic_violations(frame: pd.DataFrame, pred_col: str, score_col: str, group_cols: Optional[List[str]] = None) -> int:
    """Cuántos pares consecutivos de score suben la tasa predicha dentro de un mismo perfil.

    Con restricción monótona decreciente, al subir el score (todo lo demás igual) la tasa
    predicha nunca debe subir. Devuelve el número de violaciones (0 = se cumple).
    """
    group_cols = group_cols or []
    d = frame.sort_values(group_cols + [score_col])
    diffs = d.groupby(group_cols, observed=True)[pred_col].diff() if group_cols else d[pred_col].diff()
    return int((diffs > 1e-12).sum())


def metrics_table(frame: pd.DataFrame, model: str, fn, sets: List[str], by: List[str], **cols) -> pd.DataFrame:
    """Tabla de métricas por conjunto: total y por cada columna de `by` (estrategia, edad…).

    Parámetros:
        frame: tabla de modelado con la columna CONJUNTO y las columnas de `cols`.
        model: nombre del modelo (se guarda en la columna "modelo").
        fn: regression_metrics o rate_metrics.
        sets: conjuntos a evaluar (por ejemplo ["validacion", "prueba"]).
        by: columnas por las que también se desglosa.
        cols: argumento de fn → columna de frame (y=..., pred=..., w=... o events=..., ...).

    Devuelve:
        DataFrame con columnas modelo, conjunto, desglose, grupo y las métricas.
    """
    rows = []
    for split in sets:
        part = frame[frame["CONJUNTO"] == split]
        if part.empty:
            continue
        rows.append({"modelo": model, "conjunto": split, "desglose": "Total", "grupo": "Total",
                     **fn(**{a: part[c].to_numpy() for a, c in cols.items()})})
        for column in by:
            for key, g in part.groupby(column, observed=True):
                rows.append({"modelo": model, "conjunto": split, "desglose": column, "grupo": str(key),
                             **fn(**{a: g[c].to_numpy() for a, c in cols.items()})})
    return pd.DataFrame(rows)
