"""04 — Madurez de cohortes, curvas de Over30 y conjuntos de modelado.

Objetivo:
    Comprobar qué vintages son maduras a H = 6 (deben ser las 27 de 202401 a 202603),
    describir las curvas de la tasa acumulada de Over30 por MOB, medir con qué
    frecuencia OVER30ACCTS > ENRACCTS en MOB = 6 y construir los conjuntos de S1 y S3.

Entradas:
    - data/processed/base_estandar.csv (de 01_carga_y_diccionario.py).
    - configs/config.json: horizon_mob y data_cutoff_vintage.
    - src/data/cohorts.py.

Salidas:
    - data/processed/s1_dataset.csv: MOB = 0, todas las vintages; objetivo LINEA_PROMEDIO.
    - data/processed/s3_dataset.csv: MOB = 6, solo cohortes maduras y evaluables en
      riesgo; objetivo TASA_OVER30 = OVER30ACCTS / ENRACCTS.
    - reports/01_datos/madurez_vintages.csv
    - reports/01_datos/curvas_over30_vintage.csv y curvas_over30_estrategia.csv
    - reports/01_datos/over30_mayor_enr_mob6.csv (total y por estrategia)
    - reports/01_datos/graficas/curvas_*.html (se omiten con --sin-graficas)

Cómo correrlo (desde la raíz del repo):
    python scripts/01_datos/04_madurez_y_curvas.py
    python scripts/01_datos/04_madurez_y_curvas.py --sin-graficas

Listo cuando:
    Las vintages maduras son exactamente 202401–202603 (27), ambos datasets están
    generados y queda medida la proporción de OVER30ACCTS > ENRACCTS en MOB = 6.
    Si las vintages maduras no son las esperadas, sale con código 1.
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from src.data import catalog as cat  # noqa: E402
from src.data.cohorts import (  # noqa: E402
    build_s1_dataset, build_s3_dataset, maturity_table, over30_curves, over30_exceeds_enr,
)
from src.data.load import load_standard  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402
from src.utils.charts import STRATEGY_COLORS, save_html, style  # noqa: E402

# Vintages maduras esperadas a H = 6 con corte 202609 (CLAUDE.md §5).
EXPECTED_MATURE_FIRST, EXPECTED_MATURE_LAST, EXPECTED_MATURE_N = 202401, 202603, 27

# Proporción de OVER30ACCTS > ENRACCTS en MOB = H a partir de la cual se propone otro denominador.
FREQUENT_THRESHOLD = 0.01


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Comprueba la madurez de las vintages a H, calcula las curvas de Over30 por MOB, "
            "mide OVER30ACCTS > ENRACCTS en MOB = H y genera s1_dataset.csv y s3_dataset.csv."
        )
    )
    parser.add_argument("--sin-graficas", action="store_true", help="No genera las gráficas HTML (solo tablas y datasets).")
    return parser.parse_args()


# ===== 2. Cálculo =====
def compute(config: dict) -> dict:
    """Calcula madurez, curvas, la comprobación del denominador y los dos datasets."""
    horizon = config["horizon_mob"]
    cutoff = config["data_cutoff_vintage"]
    df = load_standard(get_path(config, "processed_dir") / "base_estandar.csv")

    maturity = maturity_table(df, horizon, cutoff)
    mature = maturity.loc[maturity["madura"], "VINTAGE"].tolist()

    # Qué hace: proporción de cohortes con OVER30ACCTS > ENRACCTS en MOB = H.
    # Por qué: el numerador es acumulado y el denominador es el saldo vigente del mes; si
    # el numerador lo supera seguido, la tasa saldría > 1 y convendría usar ACCOUNTS.
    exceeds_total = over30_exceeds_enr(df, horizon)
    exceeds_by_strategy = over30_exceeds_enr(df, horizon, ["STRATEGY"])
    exceeds = pd.concat([exceeds_total.assign(STRATEGY="Total"), exceeds_by_strategy], ignore_index=True)

    s3 = build_s3_dataset(df, horizon, cutoff)
    # Tasa de S3 con los dos denominadores posibles, como referencia para la decisión.
    rate_enr = s3["OVER30ACCTS"].sum() / s3["ENRACCTS"].sum()
    rate_accounts = s3["OVER30ACCTS"].sum() / s3["ACCOUNTS"].sum()

    return {
        "horizon": horizon,
        "maturity": maturity,
        "mature": mature,
        "curves_vintage": over30_curves(df, ["VINTAGE", "STRATEGY"]),
        "curves_strategy": over30_curves(df, ["STRATEGY"]),
        "exceeds": exceeds,
        "s1": build_s1_dataset(df),
        "s3": s3,
        "rate_enr": rate_enr,
        "rate_accounts": rate_accounts,
        "s3_rows_possible": int(((df["MOB"] == horizon) & ~df["STRATEGY"].astype(str).isin(cat.NOT_RISK_EVALUABLE)).sum()),
    }


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda los datasets en data/processed/ y las tablas en reports/01_datos/."""
    processed = get_path(config, "processed_dir")
    reports = get_path(config, "reports_dir") / "01_datos"
    reports.mkdir(parents=True, exist_ok=True)
    outputs = [
        (results["s1"], processed / "s1_dataset.csv"),
        (results["s3"], processed / "s3_dataset.csv"),
        (results["maturity"], reports / "madurez_vintages.csv"),
        (results["curves_vintage"], reports / "curvas_over30_vintage.csv"),
        (results["curves_strategy"], reports / "curvas_over30_estrategia.csv"),
        (results["exceeds"], reports / "over30_mayor_enr_mob6.csv"),
    ]
    for table, path in outputs:
        table.to_csv(path, index=False, encoding="utf-8")
    return [path for _, path in outputs]


# ===== 4. Gráficas =====
def chart_curves_by_strategy(curves: pd.DataFrame, horizon: int) -> go.Figure:
    """Líneas: tasa acumulada de Over30 (sobre ENRACCTS) por MOB, una por estrategia."""
    fig = go.Figure()
    for strategy in cat.STRATEGIES:
        s = curves[curves["STRATEGY"] == strategy]
        fig.add_scatter(
            x=s["MOB"], y=s["tasa_over30_enr"], mode="lines+markers", name=strategy,
            line_color=STRATEGY_COLORS[strategy], marker_size=8,
            hovertemplate=strategy + "<br>MOB %{x}: %{y:.2%}<extra></extra>",
        )
    fig.add_vline(x=horizon, line_dash="dot", line_color="#8a8984", annotation_text=f"H = {horizon}")
    fig.update_yaxes(tickformat=".0%")
    return style(fig, "Tasa acumulada de Over30 por MOB y estrategia", "MOB (meses desde el booking)", "Over30 / cuentas con saldo")


def chart_curves_by_vintage(curves: pd.DataFrame, horizon: int) -> go.Figure:
    """Líneas: una curva por vintage hasta MOB 12, coloreada por su estrategia."""
    fig = go.Figure()
    shown = set()
    for (vintage, strategy), s in curves[curves["MOB"] <= 12].groupby(["VINTAGE", "STRATEGY"], observed=True):
        fig.add_scatter(
            x=s["MOB"], y=s["tasa_over30_enr"], mode="lines", name=strategy, legendgroup=strategy,
            showlegend=strategy not in shown, line_color=STRATEGY_COLORS[strategy], opacity=0.75,
            hovertemplate=f"Vintage {vintage} ({strategy})<br>MOB %{{x}}: %{{y:.2%}}<extra></extra>",
        )
        shown.add(strategy)
    fig.add_vline(x=horizon, line_dash="dot", line_color="#8a8984", annotation_text=f"H = {horizon}")
    fig.update_yaxes(tickformat=".0%")
    return style(fig, "Curvas de Over30 por vintage (MOB 0 a 12)", "MOB (meses desde el booking)", "Over30 / cuentas con saldo")


def chart_rate_at_horizon(curves: pd.DataFrame, horizon: int) -> go.Figure:
    """Barras: tasa de Over30 en MOB = H por vintage madura, coloreada por estrategia."""
    at_h = curves[curves["MOB"] == horizon]
    fig = go.Figure()
    for strategy in cat.STRATEGIES:
        s = at_h[at_h["STRATEGY"] == strategy]
        if len(s):
            fig.add_bar(
                x=s["VINTAGE"].astype(str), y=s["tasa_over30_enr"], name=strategy, marker_color=STRATEGY_COLORS[strategy],
                hovertemplate="Vintage %{x}<br>" + strategy + ": %{y:.2%}<extra></extra>",
            )
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category")
    fig.update_yaxes(tickformat=".0%")
    return style(fig, f"Tasa de Over30 a {horizon} meses por vintage", "Vintage", "Over30 / cuentas con saldo")


def make_charts(results: dict, config: dict) -> list:
    """Construye las gráficas con Plotly y las guarda en reports/01_datos/graficas/."""
    out = get_path(config, "reports_dir") / "01_datos" / "graficas"
    h = results["horizon"]
    figures = {
        "curvas_over30_estrategia": chart_curves_by_strategy(results["curves_strategy"], h),
        "curvas_over30_vintage": chart_curves_by_vintage(results["curves_vintage"], h),
        "curvas_tasa_mob6_por_vintage": chart_rate_at_horizon(results["curves_vintage"], h),
    }
    return [save_html(fig, out / f"{name}.html") for name, fig in figures.items()]


# ===== 5. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 1 si las vintages maduras no son las esperadas."""
    args = parse_args()
    config = load_config()
    set_seed(config["seed"])

    results = compute(config)
    written = save_results(results, config)
    if not args.sin_graficas:
        written += make_charts(results, config)

    mature, h = results["mature"], results["horizon"]
    ok_mature = (len(mature) == EXPECTED_MATURE_N and mature[0] == EXPECTED_MATURE_FIRST and mature[-1] == EXPECTED_MATURE_LAST)
    print(f"Madurez a H = {h}:")
    print(f"  Vintages maduras: {len(mature)} ({mature[0]} a {mature[-1]}) → {'como se esperaba' if ok_mature else 'NO coincide con 202401–202603 (27)'}")

    ex = results["exceeds"]
    total = ex[ex["STRATEGY"] == "Total"].iloc[0]
    print(f"OVER30ACCTS > ENRACCTS en MOB = {h}: {int(total['cohortes_over30_mayor_enr']):,} de {int(total['cohortes']):,} cohortes ({total['proporcion']:.2%})")
    if total["proporcion"] >= FREQUENT_THRESHOLD:
        print("  Es frecuente: hay que decidir si el denominador de S3 pasa a ACCOUNTS (ver CLAUDE.md §6).")
    else:
        print("  No es frecuente: se mantiene ENRACCTS como denominador de S3.")
    print(f"  Tasa ponderada en MOB = {h}: {results['rate_enr']:.4f} sobre ENRACCTS; {results['rate_accounts']:.4f} sobre ACCOUNTS (referencia)")

    s1, s3 = results["s1"], results["s3"]
    print(f"s1_dataset.csv: {len(s1):,} cohortes, {s1['VINTAGE'].nunique()} vintages, {int(s1['ACCOUNTS'].sum()):,} cuentas")
    print(f"s3_dataset.csv: {len(s3):,} cohortes, {s3['VINTAGE'].nunique()} vintages, {int(s3['ENRACCTS'].sum()):,} cuentas con saldo"
          f" (filas descartadas por ENRACCTS = 0: {results['s3_rows_possible'] - len(s3)})")
    print("  Cohortes de S3 por estrategia: " + ", ".join(f"{k}: {v:,}" for k, v in s3["STRATEGY"].astype(str).value_counts().reindex(cat.STRATEGIES).dropna().astype(int).items()))
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0 if ok_mature else 1


if __name__ == "__main__":
    sys.exit(main())
