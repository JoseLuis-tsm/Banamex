"""S4: comparación de estrategias por perfil en la ventana de asignación aleatoria.

Perfil de comparación = PRODUCT × CHANNEL × AGE_TAG: las dimensiones que comparten las dos
familias (CLAUDE.md §5). El segmento y el score no se usan para definir el perfil porque
cada familia tiene los suyos.

Diferencia = tasa(DARIA 1.6) − tasa(r.Reca champ). Positiva: DARIA tiene más impago en ese
perfil (conviene r.Reca champ); negativa: DARIA tiene menos impago (conviene DARIA 1.6).
"""

from typing import Dict, List

import numpy as np
import pandas as pd

CHAMPION = "r.Reca champ"
CHALLENGER = "DARIA 1.6"
PROFILE_COLUMNS: List[str] = ["PRODUCT", "CHANNEL", "AGE_TAG"]
WINDOW = (202511, 202607)
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95
MIN_COHORTS = 2  # con una sola cohorte el bootstrap no tiene variación: intervalo no estimable

STATUS_COMMON = "soporte común"
STATUS_NO_EVIDENCE = "sin evidencia"


def profile_id(df: pd.DataFrame) -> pd.Series:
    """Identificador legible del perfil: 'PRODUCT | CHANNEL | AGE_TAG'."""
    return df[PROFILE_COLUMNS].astype(str).agg(" | ".join, axis=1)


def build_window(s3_dataset: pd.DataFrame, window=WINDOW) -> pd.DataFrame:
    """Cohortes maduras (MOB = 6) de la ventana aleatoria, solo champion y retadora.

    Agrega PERFIL y SOPORTE_COMUN (True si el perfil tiene cohortes con cuentas expuestas en
    las dos estrategias). DARIA 2.0 queda fuera: no es evaluable en riesgo.
    """
    w = s3_dataset[s3_dataset["VINTAGE"].between(*window) & s3_dataset["STRATEGY"].isin([CHAMPION, CHALLENGER])].copy()
    w["PERFIL"] = profile_id(w)
    with_exposure = w[w["ENRACCTS"] > 0]
    n_strategies = with_exposure.groupby("PERFIL")["STRATEGY"].nunique()
    w["SOPORTE_COMUN"] = w["PERFIL"].map(n_strategies).fillna(0).eq(2)
    return w.reset_index(drop=True)


def profile_table(window: pd.DataFrame) -> pd.DataFrame:
    """Por perfil y estrategia: cohortes, cuentas expuestas, eventos y tasa ponderada.

    Fórmula: tasa = Σ OVER30ACCTS / Σ ENRACCTS sobre las cohortes del perfil y la estrategia.
    """
    g = window.groupby(["PERFIL"] + PROFILE_COLUMNS + ["STRATEGY"]).agg(
        cohortes=("ENRACCTS", "size"), cuentas_expuestas=("ENRACCTS", "sum"), eventos=("OVER30ACCTS", "sum"),
    ).reset_index()
    g["tasa"] = g["eventos"] / g["cuentas_expuestas"]
    wide = g.pivot_table(index=["PERFIL"] + PROFILE_COLUMNS, columns="STRATEGY",
                         values=["cohortes", "cuentas_expuestas", "eventos", "tasa"])
    wide.columns = [f"{m}_{'champ' if s == CHAMPION else 'daria'}" for m, s in wide.columns]
    wide = wide.reset_index()
    for side in ("champ", "daria"):
        for m in ("cohortes", "cuentas_expuestas", "eventos"):
            col = f"{m}_{side}"
            wide[col] = wide[col].fillna(0).astype(int) if col in wide else 0
    wide["estado"] = np.where((wide["cuentas_expuestas_champ"] > 0) & (wide["cuentas_expuestas_daria"] > 0), STATUS_COMMON, STATUS_NO_EVIDENCE)
    return wide


def _bootstrap_rates(events: np.ndarray, exposed: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Tasas ponderadas de n_boot remuestreos con reemplazo de las cohortes.

    Fórmula: en cada remuestreo b, con k_bi = veces que sale la cohorte i,
    tasa_b = Σ k_bi·e_i / Σ k_bi·n_i.
    """
    n = len(events)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    return (counts @ events) / (counts @ exposed)


def bootstrap_differences(window: pd.DataFrame, seed: int, n_boot: int = N_BOOTSTRAP, level: float = CI_LEVEL) -> pd.DataFrame:
    """Método 1: diferencia observada de tasas por perfil con intervalo bootstrap por cohortes.

    Para cada perfil del soporte común, remuestrea con reemplazo las cohortes de cada
    estrategia por separado (n_boot veces, un solo generador con semilla `seed` recorriendo
    los perfiles en orden alfabético) y toma los percentiles (1 − level)/2 y (1 + level)/2 de
    la diferencia tasa_DARIA − tasa_champ.

    Devuelve por perfil: diferencia, limites del intervalo, si es estimable (al menos
    MIN_COHORTS cohortes por lado) y si es concluyente (estimable y el intervalo excluye 0).
    """
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (1 - level) / 2, (1 + level) / 2
    rows = []
    common = window[window["SOPORTE_COMUN"]]
    for perfil, g in sorted(common.groupby("PERFIL"), key=lambda kv: kv[0]):
        sides: Dict[str, pd.DataFrame] = {s: g[g["STRATEGY"] == s] for s in (CHAMPION, CHALLENGER)}
        rate = {s: d["OVER30ACCTS"].sum() / d["ENRACCTS"].sum() for s, d in sides.items()}
        diff = rate[CHALLENGER] - rate[CHAMPION]
        n_c, n_d = len(sides[CHAMPION]), len(sides[CHALLENGER])
        estimable = min(n_c, n_d) >= MIN_COHORTS
        if estimable:
            boot = {s: _bootstrap_rates(d["OVER30ACCTS"].to_numpy(float), d["ENRACCTS"].to_numpy(float), rng, n_boot) for s, d in sides.items()}
            lo, hi = np.quantile(boot[CHALLENGER] - boot[CHAMPION], [lo_q, hi_q])
        else:
            lo = hi = np.nan
        rows.append({
            "PERFIL": perfil, "cohortes_champ": n_c, "cohortes_daria": n_d,
            "tasa_champ": rate[CHAMPION], "tasa_daria": rate[CHALLENGER], "diferencia": diff,
            "ic_inferior": lo, "ic_superior": hi, "estimable": estimable,
            "concluyente": bool(estimable and (lo > 0 or hi < 0)),
        })
    return pd.DataFrame(rows)
