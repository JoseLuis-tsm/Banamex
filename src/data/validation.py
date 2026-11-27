"""Validación del contrato de datos con pandas puro (CLAUDE.md §5, "Reglas de validación").

validate() recibe un DataFrame con nombres reales (el que devuelve load_raw o
load_standard) y regresa un reporte con una fila por regla: si pasó, cuántas filas
fallaron, ejemplos de fila, qué falló y qué hacer. Las reglas bloqueantes impiden
usar el archivo; los avisos solo se informan.

Número de fila: se reporta la línea del CSV (1 = encabezado, 2 = primera fila de datos),
calculada como posición en el DataFrame + 2.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data import catalog as cat

BLOCKING = "bloqueante"
WARNING = "aviso"

# Tolerancia para las igualdades contables: |a − b| ≤ ATOL + RTOL·|b|.
ATOL = 1e-6
RTOL = 1e-9

# Cuántas filas de ejemplo se listan por regla.
MAX_EXAMPLES = 10

REPORT_COLUMNS = ["regla", "severidad", "estado", "filas_afectadas", "ejemplo_filas", "que_fallo", "que_hacer"]


def _result(rule: str, severity: str, mask: Optional[np.ndarray], what: str, action: str, count: Optional[int] = None) -> Dict[str, object]:
    """Arma la fila del reporte de una regla a partir de la máscara de filas que fallan."""
    if mask is None:
        n, examples = (count or 0), ""
    else:
        positions = np.flatnonzero(mask)
        n = int(positions.size) if count is None else count
        examples = ", ".join(str(p + 2) for p in positions[:MAX_EXAMPLES])
    return {
        "regla": rule,
        "severidad": severity,
        "estado": "pasa" if n == 0 else ("FALLA" if severity == BLOCKING else "avisa"),
        "filas_afectadas": n,
        "ejemplo_filas": examples,
        "que_fallo": what if n else "",
        "que_hacer": action if n else "",
    }


def _close(a: pd.Series, b: pd.Series) -> np.ndarray:
    """True donde a y b coinciden dentro de la tolerancia (ATOL + RTOL·|b|)."""
    return np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=RTOL, atol=ATOL)


def cohort_step_diffs(df: pd.DataFrame, column: str) -> pd.Series:
    """Cambio de `column` respecto al MOB anterior dentro de la misma cohorte.

    Fórmula: Δ_t = x_t − x_{t−1}, con t el MOB y la cohorte definida por COHORT_COLUMNS.
    Devuelve una Serie alineada con df.index; NaN en el primer MOB de cada cohorte.
    """
    ordered = df.sort_values(cat.COHORT_COLUMNS + ["MOB"], kind="mergesort")
    diffs = ordered.groupby(cat.COHORT_COLUMNS, observed=True, sort=False)[column].diff()
    return diffs.reindex(df.index)


def cumulative_evidence(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Para cada columna, qué tan seguido sube, baja o se queda igual al subir el MOB.

    Conclusión:
      - "acumulada": nunca baja y sí sube (serie no decreciente dentro de la cohorte);
      - "constante en la cohorte": nunca cambia;
      - "del mes (no acumulada)": baja en al menos un paso.
    """
    rows = []
    for col in columns:
        d = cohort_step_diffs(df, col).dropna()
        down = int((d < -ATOL).sum())
        up = int((d > ATOL).sum())
        same = int(len(d) - down - up)
        if down == 0 and up == 0:
            verdict = "constante en la cohorte"
        elif down == 0:
            verdict = "acumulada"
        else:
            verdict = "del mes (no acumulada)"
        rows.append({
            "columna": col,
            "pasos_evaluados": int(len(d)),
            "pasos_que_bajan": down,
            "pasos_que_suben": up,
            "pasos_iguales": same,
            "pct_bajan": round(100 * down / len(d), 2) if len(d) else 0.0,
            "pct_suben": round(100 * up / len(d), 2) if len(d) else 0.0,
            "conclusion": verdict,
            "se_esperaba_acumulada": col in cat.EXPECTED_CUMULATIVE,
        })
    return pd.DataFrame(rows)


def _catalog_rules(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Categorías fuera de catálogo (las de segmento y score se revisan dentro de su familia)."""
    out = []
    shared = {"STRATEGY": cat.STRATEGIES, "PRODUCT": cat.PRODUCTS, "CHANNEL": cat.CHANNELS, "AGE_TAG": cat.AGE_TAGS}
    for col, allowed in shared.items():
        mask = ~df[col].astype(str).isin(allowed).to_numpy()
        bad = sorted(df.loc[mask, col].astype(str).unique())[:5]
        out.append(_result(
            f"categoria_fuera_de_catalogo:{col}", BLOCKING, mask,
            f"{col} con valores que no están en el diccionario: {bad}",
            "Corrige el valor en el archivo o, si es una categoría nueva legítima, agrégala al diccionario y a src/data/catalog.py.",
        ))

    # Segmento y score: el par (familia, valor) debe existir en el catálogo de la familia.
    family = df["STRATEGY"].astype(str).map(cat.STRATEGY_FAMILY)
    known = family.notna().to_numpy()  # las estrategias desconocidas ya fallan arriba
    for col, by_family in (("STRATEGY_SEGMENT", cat.SEGMENTS_BY_FAMILY), ("SCORE_TAG", cat.SCORES_BY_FAMILY)):
        valid = pd.MultiIndex.from_tuples([(f, v) for f, vals in by_family.items() for v in vals])
        pairs = pd.MultiIndex.from_arrays([family.to_numpy(), df[col].astype(str).to_numpy()])
        mask = known & ~pairs.isin(valid)
        bad = sorted({f"{f}/{v}" for f, v in pairs[mask][:1000]})[:5]
        out.append(_result(
            f"categoria_fuera_de_catalogo:{col}", BLOCKING, mask,
            f"{col} que no corresponde a la familia de su estrategia: {bad}",
            "Revisa que el segmento o rango de score sea el de la familia de la estrategia (Recalibration o DARIA).",
        ))
    return out


def _range_rules(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Valores no numéricos, fuera del rango del diccionario o no enteros donde deben serlo."""
    out = []
    ranges = {"VINTAGE": (cat.VINTAGE_MIN, cat.VINTAGE_MAX), "MOB": (cat.MOB_MIN, cat.MOB_MAX)}
    for col in cat.VOLUME_COLUMNS + cat.RESULT_COLUMNS:
        ranges[col] = (cat.AMOUNT_MIN, cat.AMOUNT_MAX)
    for col, (lo, hi) in ranges.items():
        values = pd.to_numeric(df[col], errors="coerce")
        mask = (values.isna() | (values < lo) | (values > hi)).to_numpy()
        if col in cat.INTEGER_COLUMNS:
            mask |= (values.notna() & (values != np.floor(values))).to_numpy()
        out.append(_result(
            f"valor_fuera_de_rango:{col}", BLOCKING, mask,
            f"{col} no numérico, fuera de [{lo}, {hi}]" + (" o no entero" if col in cat.INTEGER_COLUMNS else ""),
            "Corrige el valor en el archivo de origen; el diccionario fija ese rango.",
        ))
    return out


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas las reglas del contrato y devuelve el reporte (una fila por regla).

    Parámetros:
        df: datos con nombres reales. Las columnas que no están en el diccionario
            generan un aviso (y el cargador las descarta).

    Devuelve:
        DataFrame con columnas REPORT_COLUMNS. Si faltan columnas del contrato, solo se
        reporta esa falla (las demás reglas no se pueden evaluar).
    """
    df = df.reset_index(drop=True)
    results: List[Dict[str, object]] = []

    # ===== Estructura =====
    missing = [c for c in cat.ALL_COLUMNS if c not in df.columns]
    results.append(_result(
        "columna_faltante", BLOCKING, None, f"Faltan columnas del contrato: {missing}",
        "Agrega las columnas faltantes (con nombre real o alias dummiN).", count=len(missing),
    ))
    undeclared = [c for c in df.columns if c not in cat.ALL_COLUMNS]
    results.append(_result(
        "columna_no_declarada", WARNING, None, f"Columnas que no están en el diccionario: {undeclared}",
        "Se descartan al cargar; si son necesarias, declararlas en el diccionario.", count=len(undeclared),
    ))
    if missing:
        return pd.DataFrame(results, columns=REPORT_COLUMNS)

    nulls = df[cat.ALL_COLUMNS].isna().any(axis=1).to_numpy()
    results.append(_result(
        "valor_nulo", BLOCKING, nulls, "Filas con algún valor vacío en las columnas del contrato.",
        "Completa los valores vacíos; ninguna columna del contrato admite nulos.",
    ))
    dup = df.duplicated(cat.KEY_COLUMNS, keep=False).to_numpy()
    results.append(_result(
        "llave_repetida", BLOCKING, dup, "Filas con la misma llave (7 dimensiones + MOB).",
        "Deja una sola fila por llave (suma o elimina duplicados en el origen).",
    ))

    # ===== Catálogos y rangos =====
    results += _catalog_rules(df)
    results += _range_rules(df)

    # ===== Estrategia según vintage y MOB posible =====
    vintages = pd.to_numeric(df["VINTAGE"], errors="coerce")
    allowed = {v: cat.allowed_strategies(v) for v in vintages.dropna().unique()}
    strat = df["STRATEGY"].astype(str).to_numpy()
    ok_strategy = np.array([s in (allowed.get(v) or ()) for v, s in zip(vintages.to_numpy(), strat)])
    results.append(_result(
        "estrategia_no_corresponde_a_vintage", BLOCKING, ~ok_strategy,
        "La estrategia no es la que corresponde a la vintage según el diccionario.",
        "Revisa la vintage o la estrategia; ver tabla 'Estrategia según vintage' en CLAUDE.md §5.",
    ))
    mob = pd.to_numeric(df["MOB"], errors="coerce")
    max_possible = vintages.map(lambda v: cat.max_mob(v) if pd.notna(v) else np.nan)
    results.append(_result(
        "mob_mayor_al_posible", BLOCKING, (mob > max_possible).to_numpy(),
        "MOB mayor que los meses entre la vintage y el corte 202609.",
        "Revisa la vintage o el MOB; con corte 202609 el MOB máximo es 202609 − vintage en meses.",
    ))

    # ===== Igualdades contables =====
    ncl_bad = ~_close(df["NCLAMTS"], df["GCLAMTS"] - df["RECOVERYAMTS"])
    results.append(_result(
        "ncl_distinto_de_gcl_menos_recuperacion", BLOCKING, ncl_bad,
        "NCLAMTS ≠ GCLAMTS − RECOVERYAMTS.", "Recalcula NCLAMTS como GCLAMTS − RECOVERYAMTS en el origen.",
    ))
    total = df["EXPENSES_ACQ"] + df["EXPENSES_ECM_VAR"] + df["EXPENSES_ECM_FIX"]
    results.append(_result(
        "expenses_total_distinto_de_la_suma", BLOCKING, ~_close(df["EXPENSES_TOTAL"], total),
        "EXPENSES_TOTAL ≠ EXPENSES_ACQ + EXPENSES_ECM_VAR + EXPENSES_ECM_FIX.",
        "Recalcula EXPENSES_TOTAL como la suma de sus tres componentes.",
    ))
    acq_bad = ((mob != 0) & ~_close(df["EXPENSES_ACQ"], pd.Series(0.0, index=df.index))).to_numpy()
    results.append(_result(
        "expenses_acq_con_mob_distinto_de_0", BLOCKING, acq_bad,
        "EXPENSES_ACQ distinto de 0 en un MOB distinto de 0.",
        "El gasto de adquisición solo se registra en MOB = 0; corrige el origen.",
    ))

    # ===== Avisos =====
    results.append(_result(
        "over30_mayor_que_enr", WARNING, (df["OVER30ACCTS"] > df["ENRACCTS"]).to_numpy(),
        "OVER30ACCTS (acumulado) mayor que ENRACCTS (cuentas con saldo en el mes).",
        "No bloquea; afecta la tasa de S3 si es frecuente en MOB = 6 (ver 04_madurez_y_curvas.py).",
    ))
    for col in cat.EXPECTED_CUMULATIVE:
        down = (cohort_step_diffs(df, col) < -ATOL).to_numpy()
        results.append(_result(
            f"acumulada_baja:{col}", WARNING, down,
            f"{col} es acumulada pero baja respecto al MOB anterior de la misma cohorte.",
            "Revisa el origen: una serie acumulada no debería bajar al subir el MOB.",
        ))

    return pd.DataFrame(results, columns=REPORT_COLUMNS)


def has_blocking_failures(report: pd.DataFrame) -> bool:
    """True si alguna regla bloqueante falló."""
    return bool(((report["severidad"] == BLOCKING) & (report["filas_afectadas"] > 0)).any())
