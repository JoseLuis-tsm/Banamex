"""05 — Balance entre estrategias en la ventana de asignación aleatoria.

Objetivo:
    Aislar la ventana de asignación aleatoria 202511–202607 y comparar la mezcla de
    perfiles entre r.Reca champ y DARIA 1.6 en las dimensiones compartidas (edad,
    producto, canal y posición de score). Medir la proporción de cuentas asignada a cada
    estrategia por vintage y marcar DARIA 2.0 como no evaluable en riesgo.

Entradas:
    - data/processed/base_estandar.csv (de 01_carga_y_diccionario.py).
    - src/strategy/balance.py, src/data/catalog.py.

Salidas (reports/01_datos/):
    - balance_estrategias.csv: por variable y categoría, participación en cada estrategia,
      diferencia en puntos porcentuales, SMD y si hay desbalance (|SMD| > 0.10).
    - balance_asignacion_por_vintage.csv: proporción de cuentas por estrategia en cada
      vintage de asignación aleatoria, contra la proporción esperada.
    - estrategias_evaluables.csv: por estrategia, vintages, MOB máximo observado y si es
      evaluable en riesgo a H.
    - graficas/balance_*.html (se omiten con --sin-graficas)

    Solo se usa MOB = 0 (una fila por cohorte, peso ACCOUNTS). Los segmentos no se
    comparan: cada familia tiene los suyos.

Cómo correrlo (desde la raíz del repo):
    python scripts/01_datos/05_balance_entre_estrategias.py
    python scripts/01_datos/05_balance_entre_estrategias.py --sin-graficas

Listo cuando:
    El diagnóstico de balance queda listo para que lo use S4 en la Fase 3.
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
from src.strategy.balance import SMD_THRESHOLD, categorical_balance, continuous_balance  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402
from src.utils.charts import STRATEGY_COLORS, TEXT_SECONDARY, save_html, style  # noqa: E402

CHAMPION = "r.Reca champ"
CHALLENGER = "DARIA 1.6"
WINDOW = (202511, 202607)
# Proporción esperada de cuentas para la retadora en cada ventana aleatoria (diccionario).
EXPECTED_SHARE = {"DARIA 1.6": 0.30, "DARIA 2.0": 0.50}


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Compara la mezcla de perfiles entre r.Reca champ y DARIA 1.6 en la ventana de "
            "asignación aleatoria (202511–202607), mide la asignación por vintage y marca "
            "las estrategias no evaluables en riesgo."
        )
    )
    parser.add_argument("--sin-graficas", action="store_true", help="No genera las gráficas HTML (solo tablas).")
    return parser.parse_args()


# ===== 2. Cálculo =====
def add_score_features(booked: pd.DataFrame) -> pd.DataFrame:
    """Agrega la posición relativa del score (0–1 en su familia) y su cuartil."""
    family = booked["STRATEGY"].astype(str).map(cat.STRATEGY_FAMILY)
    booked = booked.assign(
        SCORE_POSICION=[cat.score_position(s, f) for s, f in zip(booked["SCORE_TAG"].astype(str), family)]
    )
    # Cuartil de la posición: [0, .25], (.25, .5], (.5, .75], (.75, 1]
    booked["SCORE_CUARTIL"] = pd.cut(
        booked["SCORE_POSICION"], bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
        labels=["Q1 (score bajo)", "Q2", "Q3", "Q4 (score alto)"],
    )
    return booked


def assignment_by_vintage(booked: pd.DataFrame) -> pd.DataFrame:
    """Proporción de cuentas por estrategia en cada vintage aleatoria, contra la esperada."""
    random = booked[booked["VINTAGE"].map(cat.assignment_type) == "aleatoria"]
    t = random.groupby(["VINTAGE", "STRATEGY"], observed=True)["ACCOUNTS"].sum().reset_index(name="cuentas")
    t["proporcion_observada"] = t["cuentas"] / t.groupby("VINTAGE")["cuentas"].transform("sum")
    challenger = t["VINTAGE"].map(lambda v: [s for s in cat.allowed_strategies(v) if s != CHAMPION][0])
    t["proporcion_esperada"] = [
        EXPECTED_SHARE[c] if s == c else 1 - EXPECTED_SHARE[c] for s, c in zip(t["STRATEGY"].astype(str), challenger)
    ]
    t["diferencia_pp"] = 100 * (t["proporcion_observada"] - t["proporcion_esperada"])
    return t


def evaluable_strategies(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Por estrategia: familia, vintages, MOB máximo observado y si es evaluable en riesgo."""
    t = df.groupby("STRATEGY", observed=True).agg(
        vintage_inicial=("VINTAGE", "min"), vintage_final=("VINTAGE", "max"), mob_max_observado=("MOB", "max"),
    ).reindex(cat.STRATEGIES).reset_index()
    t["familia"] = t["STRATEGY"].map(cat.STRATEGY_FAMILY)
    t["evaluable_en_riesgo"] = ~t["STRATEGY"].isin(cat.NOT_RISK_EVALUABLE) & (t["mob_max_observado"] >= horizon)
    t["motivo"] = [
        "" if ok else f"Solo tiene MOB 0 a {m}: tiene línea asignada pero no desempeño de riesgo a {horizon} meses."
        for ok, m in zip(t["evaluable_en_riesgo"], t["mob_max_observado"])
    ]
    return t


def compute(config: dict) -> dict:
    """Calcula el balance de perfiles, la asignación por vintage y las estrategias evaluables."""
    df = load_standard(get_path(config, "processed_dir") / "base_estandar.csv")
    booked = add_score_features(df[df["MOB"] == 0].copy())
    window = booked[booked["VINTAGE"].between(*WINDOW)]
    window = window.assign(STRATEGY=window["STRATEGY"].astype(str))

    # Qué hace: para cada dimensión compartida, compara la participación ponderada por
    # ACCOUNTS en champion (a) y retadora (b) y calcula la SMD. Por qué: con asignación
    # aleatoria la mezcla debería ser igual; un desbalance obliga a ajustar por perfil en S4.
    parts = [categorical_balance(window, col, "STRATEGY", CHAMPION, CHALLENGER) for col in ("AGE_TAG", "PRODUCT", "CHANNEL", "SCORE_CUARTIL")]
    parts.append(continuous_balance(window, "SCORE_POSICION", "STRATEGY", CHAMPION, CHALLENGER))
    balance = pd.concat(parts, ignore_index=True).rename(columns={
        "participacion_a": f"valor_{CHAMPION}", "participacion_b": f"valor_{CHALLENGER}",
    })

    return {
        "balance": balance,
        "assignment": assignment_by_vintage(booked),
        "evaluable": evaluable_strategies(df, config["horizon_mob"]),
        "window_accounts": window.groupby("STRATEGY")["ACCOUNTS"].sum().to_dict(),
    }


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda las tres tablas en reports/01_datos/."""
    reports = get_path(config, "reports_dir") / "01_datos"
    reports.mkdir(parents=True, exist_ok=True)
    written = []
    for key, name in (("balance", "balance_estrategias.csv"), ("assignment", "balance_asignacion_por_vintage.csv"), ("evaluable", "estrategias_evaluables.csv")):
        out = reports / name
        results[key].round(6).to_csv(out, index=False, encoding="utf-8")
        written.append(out)
    return written


# ===== 4. Gráficas =====
def chart_smd(balance: pd.DataFrame) -> go.Figure:
    """Puntos: SMD por variable y categoría, con las bandas de ±0.10."""
    b = balance.iloc[::-1]
    labels = b["variable"] + ": " + b["categoria"]
    fig = go.Figure(go.Scatter(
        x=b["smd"], y=labels, mode="markers", name="SMD",
        marker={"size": 9, "color": [STRATEGY_COLORS[CHALLENGER] if d else STRATEGY_COLORS["14.Recalibration 1.0"] for d in b["desbalance"]]},
        hovertemplate="%{y}<br>SMD: %{x:.3f}<extra></extra>",
    ))
    for x in (-SMD_THRESHOLD, SMD_THRESHOLD):
        fig.add_vline(x=x, line_dash="dot", line_color=TEXT_SECONDARY)
    fig.add_vline(x=0, line_color=TEXT_SECONDARY, line_width=1)
    fig.update_layout(height=max(500, 22 * len(b)), showlegend=False)
    return style(fig, f"Balance de perfiles: {CHAMPION} vs. {CHALLENGER} (202511–202607)",
                 f"Diferencia estandarizada (SMD; positiva = más en {CHAMPION})", "")


def chart_assignment(assign: pd.DataFrame) -> go.Figure:
    """Barras apiladas: proporción de cuentas por estrategia en cada vintage aleatoria."""
    fig = go.Figure()
    for strategy in (CHAMPION, "DARIA 1.6", "DARIA 2.0"):
        s = assign[assign["STRATEGY"] == strategy]
        fig.add_bar(
            x=s["VINTAGE"].astype(str), y=s["proporcion_observada"], name=strategy, marker_color=STRATEGY_COLORS[strategy],
            customdata=s["proporcion_esperada"],
            hovertemplate="Vintage %{x}<br>" + strategy + ": %{y:.1%} (esperado %{customdata:.0%})<extra></extra>",
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(type="category")
    fig.update_yaxes(tickformat=".0%")
    return style(fig, "Asignación de cuentas por vintage en las ventanas aleatorias", "Vintage", "% de las cuentas de la vintage")


def make_charts(results: dict, config: dict) -> list:
    """Construye las gráficas con Plotly y las guarda en reports/01_datos/graficas/."""
    out = get_path(config, "reports_dir") / "01_datos" / "graficas"
    return [
        save_html(chart_smd(results["balance"]), out / "balance_smd_perfiles.html"),
        save_html(chart_assignment(results["assignment"]), out / "balance_asignacion_por_vintage.html"),
    ]


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

    b = results["balance"]
    acc = results["window_accounts"]
    print(f"Ventana aleatoria {WINDOW[0]}–{WINDOW[1]}: {CHAMPION} {acc.get(CHAMPION, 0):,} cuentas; {CHALLENGER} {acc.get(CHALLENGER, 0):,} cuentas")
    print(f"  Categorías comparadas: {len(b)}; con |SMD| > {SMD_THRESHOLD}: {int(b['desbalance'].sum())}")
    for _, row in b.reindex(b["smd"].abs().sort_values(ascending=False).index).head(5).iterrows():
        print(f"    {row['variable']}: {row['categoria']:22s} SMD {row['smd']:+.3f}")
    a = results["assignment"]
    challengers = a[a["STRATEGY"] != CHAMPION]
    print(f"  Asignación de la retadora por vintage: desviación máxima respecto a lo esperado {challengers['diferencia_pp'].abs().max():.1f} pp")
    print("Estrategias evaluables en riesgo:")
    for _, row in results["evaluable"].iterrows():
        print(f"  {row['STRATEGY']:22s} MOB máx {row['mob_max_observado']:>2}  → {'evaluable' if row['evaluable_en_riesgo'] else 'NO evaluable: ' + row['motivo']}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
