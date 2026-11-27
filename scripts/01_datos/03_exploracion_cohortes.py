"""03 — Exploración de cohortes.

Objetivo:
    Describir la base: cuentas por vintage, estrategia y segmento; participación de
    productos y canales; distribución de score y edad por familia; filas por MOB.

Entradas:
    - data/processed/base_estandar.csv (de 01_carga_y_diccionario.py).

Salidas (reports/01_datos/):
    - exploracion_cuentas_vintage_estrategia.csv
    - exploracion_cuentas_estrategia_segmento.csv
    - exploracion_participacion_producto.csv
    - exploracion_participacion_canal.csv
    - exploracion_score_por_familia.csv
    - exploracion_edad_por_familia.csv
    - exploracion_filas_por_mob.csv
    - graficas/exploracion_*.html (se omiten con --sin-graficas)

    Las cuentas se cuentan en MOB = 0 (ACCOUNTS no cambia dentro de una cohorte), así
    cada cuenta se cuenta una sola vez.

Cómo correrlo (desde la raíz del repo):
    python scripts/01_datos/03_exploracion_cohortes.py
    python scripts/01_datos/03_exploracion_cohortes.py --sin-graficas

Listo cuando:
    Las tablas y gráficas se generan con una sola corrida.
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from src.data import catalog as cat  # noqa: E402
from src.data.load import load_standard  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402
from src.utils.charts import FAMILY_COLORS, SEQUENTIAL_BLUE, STRATEGY_COLORS, save_html, style  # noqa: E402


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Explora la base estándar: cuentas por vintage, estrategia y segmento; "
            "participación de productos y canales; score y edad por familia; filas por MOB."
        )
    )
    parser.add_argument("--sin-graficas", action="store_true", help="No genera las gráficas HTML (solo tablas).")
    return parser.parse_args()


# ===== 2. Cálculo =====
def share_table(booked: pd.DataFrame, group: str, column: str, order: list) -> pd.DataFrame:
    """Cuentas y participación (%) de cada valor de `column` dentro de cada `group`.

    Fórmula: participación = 100 · cuentas(grupo, valor) / cuentas(grupo).
    """
    t = booked.groupby([group, column], observed=True)["ACCOUNTS"].sum().reset_index(name="cuentas")
    t["participacion_pct"] = 100 * t["cuentas"] / t.groupby(group, observed=True)["cuentas"].transform("sum")
    t[column] = pd.Categorical(t[column].astype(str), categories=order, ordered=True)
    return t.sort_values([group, column]).reset_index(drop=True)


def compute(config: dict) -> dict:
    """Arma las tablas de exploración a partir de la base estándar."""
    df = load_standard(get_path(config, "processed_dir") / "base_estandar.csv")
    df["FAMILY"] = df["STRATEGY"].astype(str).map(cat.STRATEGY_FAMILY)
    # Qué hace: booked guarda solo MOB = 0, una fila por cohorte. Por qué: ACCOUNTS es
    # constante dentro de la cohorte; sumarlo en todos los MOB contaría cada cuenta varias veces.
    booked = df[df["MOB"] == 0]

    by_vintage = (
        booked.groupby(["VINTAGE", "STRATEGY"], observed=True)
        .agg(cohortes=("ACCOUNTS", "size"), cuentas=("ACCOUNTS", "sum"))
        .reset_index()
    )
    by_vintage["participacion_en_vintage_pct"] = 100 * by_vintage["cuentas"] / by_vintage.groupby("VINTAGE")["cuentas"].transform("sum")
    by_vintage["tipo_asignacion"] = by_vintage["VINTAGE"].map(cat.assignment_type)

    by_segment = (
        booked.groupby(["STRATEGY", "STRATEGY_SEGMENT"], observed=True)
        .agg(cohortes=("ACCOUNTS", "size"), cuentas=("ACCOUNTS", "sum"))
        .reset_index()
    )
    by_segment["participacion_en_estrategia_pct"] = 100 * by_segment["cuentas"] / by_segment.groupby("STRATEGY", observed=True)["cuentas"].transform("sum")

    scores = []
    for family, order in cat.SCORES_BY_FAMILY.items():
        t = share_table(booked[booked["FAMILY"] == family], "FAMILY", "SCORE_TAG", order)
        t["posicion_relativa"] = [cat.score_position(s, family) for s in t["SCORE_TAG"].astype(str)]
        scores.append(t)

    by_mob = (
        df.groupby("MOB").agg(filas=("ACCOUNTS", "size"), vintages=("VINTAGE", "nunique")).reset_index()
    )

    return {
        "cuentas_vintage_estrategia": by_vintage,
        "cuentas_estrategia_segmento": by_segment,
        "participacion_producto": share_table(booked, "STRATEGY", "PRODUCT", cat.PRODUCTS),
        "participacion_canal": share_table(booked, "STRATEGY", "CHANNEL", cat.CHANNELS),
        "score_por_familia": pd.concat(scores, ignore_index=True),
        "edad_por_familia": share_table(booked, "FAMILY", "AGE_TAG", cat.AGE_TAGS),
        "filas_por_mob": by_mob,
        "n_rows": len(df),
        "n_cohorts": len(booked),
    }


# ===== 3. Guardado =====
TABLES = [
    "cuentas_vintage_estrategia", "cuentas_estrategia_segmento", "participacion_producto",
    "participacion_canal", "score_por_familia", "edad_por_familia", "filas_por_mob",
]


def save_results(results: dict, config: dict) -> list:
    """Guarda cada tabla como reports/01_datos/exploracion_<nombre>.csv."""
    reports = get_path(config, "reports_dir") / "01_datos"
    reports.mkdir(parents=True, exist_ok=True)
    written = []
    for name in TABLES:
        out = reports / f"exploracion_{name}.csv"
        results[name].round(4).to_csv(out, index=False, encoding="utf-8")
        written.append(out)
    return written


# ===== 4. Gráficas =====
def chart_accounts_by_vintage(t: pd.DataFrame) -> go.Figure:
    """Barras apiladas: cuentas colocadas por vintage, por estrategia."""
    fig = go.Figure()
    for strategy in cat.STRATEGIES:
        s = t[t["STRATEGY"] == strategy]
        fig.add_bar(
            x=s["VINTAGE"].astype(str), y=s["cuentas"], name=strategy, marker_color=STRATEGY_COLORS[strategy],
            hovertemplate="Vintage %{x}<br>" + strategy + ": %{y:,} cuentas<extra></extra>",
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(type="category")
    return style(fig, "Cuentas colocadas por vintage y estrategia", "Vintage (mes de booking)", "Cuentas")


def chart_share_heatmap(t: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Mapa de calor: participación (%) de cada producto o canal dentro de cada estrategia."""
    pivot = t.pivot_table(index=column, columns="STRATEGY", values="participacion_pct", observed=False).reindex(columns=cat.STRATEGIES)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index.astype(str), colorscale=SEQUENTIAL_BLUE,
        text=pivot.round(1).values, texttemplate="%{text}", colorbar={"title": "%"},
        hovertemplate="%{x}<br>%{y}: %{z:.1f}% de las cuentas<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return style(fig, title, "Estrategia", "")


def chart_family_distribution(t: pd.DataFrame, column: str, title: str, x_title: str) -> go.Figure:
    """Barras agrupadas por familia: participación (%) de cada rango (score o edad)."""
    fig = go.Figure()
    for family in (cat.RECALIBRATION, cat.DARIA):
        s = t[t["FAMILY"] == family]
        fig.add_bar(
            x=s[column].astype(str), y=s["participacion_pct"], name=family, marker_color=FAMILY_COLORS[family],
            hovertemplate=family + "<br>%{x}: %{y:.1f}%<extra></extra>",
        )
    fig.update_layout(barmode="group")
    return style(fig, title, x_title, "% de las cuentas de la familia")


def chart_rows_by_mob(t: pd.DataFrame) -> go.Figure:
    """Barras: filas por MOB (decrecen porque las vintages recientes tienen menos meses observados)."""
    fig = go.Figure(go.Bar(
        x=t["MOB"], y=t["filas"], marker_color=STRATEGY_COLORS["14.Recalibration 1.0"], name="Filas",
        customdata=t["vintages"], hovertemplate="MOB %{x}: %{y:,} filas de %{customdata} vintages<extra></extra>",
    ))
    return style(fig, "Filas por MOB", "MOB (meses desde el booking)", "Filas")


def make_charts(results: dict, config: dict) -> list:
    """Construye las gráficas con Plotly y las guarda en reports/01_datos/graficas/."""
    out = get_path(config, "reports_dir") / "01_datos" / "graficas"
    figures = {
        "exploracion_cuentas_por_vintage": chart_accounts_by_vintage(results["cuentas_vintage_estrategia"]),
        "exploracion_producto_por_estrategia": chart_share_heatmap(results["participacion_producto"], "PRODUCT", "Participación de producto por estrategia (%)"),
        "exploracion_canal_por_estrategia": chart_share_heatmap(results["participacion_canal"], "CHANNEL", "Participación de canal por estrategia (%)"),
        "exploracion_score_por_familia": chart_family_distribution(
            results["score_por_familia"].sort_values(["FAMILY", "posicion_relativa"]), "SCORE_TAG",
            "Distribución de score por familia (cada familia tiene su propia escala)", "Rango de score"),
        "exploracion_edad_por_familia": chart_family_distribution(results["edad_por_familia"], "AGE_TAG", "Distribución de edad por familia", "Rango de edad"),
        "exploracion_filas_por_mob": chart_rows_by_mob(results["filas_por_mob"]),
    }
    return [save_html(fig, out / f"{name}.html") for name, fig in figures.items()]


# ===== 5. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 0 si todo salió bien."""
    args = parse_args()
    config = load_config()
    set_seed(config["seed"])

    results = compute(config)
    written = save_results(results, config)
    if not args.sin_graficas:
        written += make_charts(results, config)

    v = results["cuentas_vintage_estrategia"]
    print(f"Exploración de {results['n_rows']:,} filas y {results['n_cohorts']:,} cohortes (MOB = 0):")
    print(f"  Cuentas colocadas: {int(v['cuentas'].sum()):,} en {v['VINTAGE'].nunique()} vintages")
    for strategy, n in v.groupby("STRATEGY", observed=True)["cuentas"].sum().reindex(cat.STRATEGIES).items():
        print(f"    {strategy:22s} {int(n):>10,} cuentas")
    seg = results["cuentas_estrategia_segmento"]
    print(f"  Segmentos con cuentas: {seg.groupby('STRATEGY', observed=True).size().reindex(cat.STRATEGIES).to_dict()}")
    print(f"  MOB observados: {results['filas_por_mob']['MOB'].min()} a {results['filas_por_mob']['MOB'].max()}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
