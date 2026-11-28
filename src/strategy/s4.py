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


def bootstrap_differences(window: pd.DataFrame, seed: int, n_boot: int = N_BOOTSTRAP, level: float = CI_LEVEL,
                          events_column: str = "OVER30ACCTS") -> pd.DataFrame:
    """Diferencia de tasas por perfil con intervalo bootstrap por cohortes.

    Método 1: events_column = OVER30ACCTS (eventos observados).
    Método 2: events_column = eventos esperados (ENRACCTS × tasa predicha), así la "tasa" del
    perfil es la tasa predicha ponderada por ENRACCTS.

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
        rate = {s: d[events_column].sum() / d["ENRACCTS"].sum() for s, d in sides.items()}
        diff = rate[CHALLENGER] - rate[CHAMPION]
        n_c, n_d = len(sides[CHAMPION]), len(sides[CHALLENGER])
        estimable = min(n_c, n_d) >= MIN_COHORTS
        if estimable:
            boot = {s: _bootstrap_rates(d[events_column].to_numpy(float), d["ENRACCTS"].to_numpy(float), rng, n_boot) for s, d in sides.items()}
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


# ===== Método 2: predicción con S3 dejando fuera una vintage a la vez =====
def leave_one_vintage_out(frame: pd.DataFrame, window: pd.DataFrame, params: dict, rounds: int, a: float, b: float) -> pd.Series:
    """Tasa predicha (calibrada) de cada cohorte de la ventana con un S3 que no la vio.

    Para cada vintage v de la ventana: entrena un XGBoost con todas las cohortes de `frame`
    salvo las de v (mismos hiperparámetros `params` y rondas `rounds` del modelo publicado,
    sin búsqueda nueva), predice las cohortes de la ventana con vintage v y aplica la misma
    calibración de Platt, σ(a·logit(p) + b).

    Por qué: el S3 publicado se entrenó con estas mismas cohortes; predecirlas con él
    inflaría la coincidencia de signo con el método 1.

    Parámetros:
        frame: cohortes de modelado de S3 (con CONJUNTO, TASA_OVER30, ENRACCTS y las variables).
        window: cohortes de la ventana (llave de 7 dimensiones + VINTAGE).
    Devuelve: Serie alineada con el índice de `window`.
    """
    import xgboost as xgb  # import local: el resto del módulo no necesita xgboost

    from src.data import catalog as cat
    from src.features.engineering import build_features, feature_columns
    from src.models.predict import platt_transform
    from src.models.tuning import train_booster

    cols = feature_columns()
    out = pd.Series(np.nan, index=window.index, dtype=float)
    for v in sorted(window["VINTAGE"].unique()):
        train = frame[frame["VINTAGE"] != v]
        dtrain = xgb.DMatrix(train[cols], label=train["TASA_OVER30"], weight=train["ENRACCTS"], feature_names=cols)
        booster = train_booster(params, dtrain, num_rounds=rounds)
        target = window[window["VINTAGE"] == v]
        p = booster.predict(xgb.DMatrix(build_features(target[cat.COHORT_COLUMNS]), feature_names=cols))
        out.loc[target.index] = platt_transform(p, a, b)
    return out


# ===== Regla de decisión y concordancia =====
NO_CONCLUSIVE = "Sin diferencia concluyente"
NO_EVIDENCE = "Sin evidencia"
NOT_RISK_EVALUABLE = "No evaluable en riesgo"


def decide(m1: pd.DataFrame, m2: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Sugerencia por perfil combinando los dos métodos.

    Regla: se sugiere una estrategia solo si el intervalo del método 1 excluye cero y la
    diferencia del método 2 tiene el mismo signo. Diferencia > 0 (DARIA con más impago) →
    r.Reca champ; < 0 → DARIA 1.6. Si no, "Sin diferencia concluyente". Los perfiles fuera
    del soporte común salen como "Sin evidencia".

    Parámetros:
        m1: salida de bootstrap_differences con eventos observados.
        m2: igual, con eventos esperados (solo se usan PERFIL y diferencia).
        profiles: profile_table (PERFIL y estado).
    Devuelve: por perfil, diferencias de ambos métodos, intervalo, sugerencia y motivo.
    """
    m = m1.merge(m2[["PERFIL", "diferencia"]].rename(columns={"diferencia": "diferencia_m2"}), on="PERFIL", how="left")
    m = m.rename(columns={"diferencia": "diferencia_m1"})
    s1, s2 = np.sign(m["diferencia_m1"]), np.sign(m["diferencia_m2"])
    m["mismo_signo"] = s1.eq(s2) & s1.ne(0)
    suggest = m["concluyente"] & m["mismo_signo"]
    m["sugerencia"] = np.where(suggest, np.where(m["diferencia_m1"] > 0, CHAMPION, CHALLENGER), NO_CONCLUSIVE)
    m["motivo"] = np.select(
        [suggest, ~m["estimable"], ~m["concluyente"]],
        ["intervalo excluye cero y los dos métodos coinciden en el signo",
         f"menos de {MIN_COHORTS} cohortes en alguna estrategia: intervalo no estimable",
         "el intervalo del método 1 incluye cero"],
        default="los métodos no coinciden en el signo",
    )
    outside = profiles.loc[profiles["estado"] == STATUS_NO_EVIDENCE, ["PERFIL"]].assign(
        sugerencia=NO_EVIDENCE, motivo="el perfil no tiene cuentas expuestas en las dos estrategias (fuera del soporte común)")
    return pd.concat([m, outside], ignore_index=True).sort_values("PERFIL").reset_index(drop=True)


def sign_agreement(decision: pd.DataFrame) -> dict:
    """Concordancia: fracción de perfiles concluyentes en el método 1 cuyo signo repite el método 2."""
    c = decision[decision["concluyente"].eq(True)]
    n = len(c)
    agree = int(c["mismo_signo"].sum())
    return {"perfiles_concluyentes": n, "coinciden": agree, "concordancia": agree / n if n else float("nan")}
