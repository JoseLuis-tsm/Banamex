"""02 — S4, método 1: diferencia observada con intervalo bootstrap.

Objetivo:
    Para cada perfil del soporte común (PRODUCT × CHANNEL × AGE_TAG), calcular la diferencia
    observada de la tasa de Over30 a 6 meses entre DARIA 1.6 y r.Reca champ, con intervalo de
    95% por bootstrap sobre cohortes (2,000 remuestreos, semilla 42).

    Diferencia = tasa(DARIA 1.6) − tasa(r.Reca champ); positiva = DARIA con más impago.
    Un perfil con menos de 2 cohortes en alguna estrategia no tiene intervalo estimable (el
    bootstrap de una sola cohorte no tiene variación) y nunca es concluyente.

Entradas:
    - data/processed/s4_ventana.csv (de 01_ventana_aleatoria.py).
    - src/strategy/s4.py.

Salidas:
    - reports/03_estrategia_y_explicabilidad/s4_metodo1_bootstrap.csv: por perfil, cohortes y
      tasa de cada estrategia, diferencia, intervalo, si es estimable y si es concluyente.
    - reports/03_estrategia_y_explicabilidad/graficas/s4_metodo1_intervalos.html

Cómo correrlo (desde la raíz del repo):
    python scripts/03_estrategia_y_explicabilidad/02_s4_observado_bootstrap.py

Listo cuando:
    Cada perfil del soporte común tiene diferencia e intervalo, y el bootstrap es
    reproducible (se comprueba corriéndolo dos veces dentro del script).
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from src.strategy.s4 import CHALLENGER, CHAMPION, CI_LEVEL, N_BOOTSTRAP, bootstrap_differences  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402
from src.utils.charts import CATEGORICAL, TEXT_SECONDARY, save_html, style  # noqa: E402


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "S4 método 1: diferencia observada de la tasa de Over30 (DARIA 1.6 − r.Reca champ) por "
            "perfil, con intervalo bootstrap de 95%% sobre cohortes (semilla 42)."
        )
    )
    parser.add_argument("--sin-graficas", action="store_true", help="No genera la gráfica HTML.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def compute(config: dict) -> dict:
    """Calcula diferencias e intervalos y comprueba que el bootstrap sea reproducible."""
    window = pd.read_csv(get_path(config, "processed_dir") / "s4_ventana.csv", encoding="utf-8")
    seed = config["seed"]
    # Qué hace: remuestrea con reemplazo las cohortes de cada estrategia dentro del perfil y
    # toma los percentiles 2.5 y 97.5 de la diferencia. Por qué: la cohorte es la unidad que
    # se observa (las cuentas de una cohorte no son independientes entre sí).
    result = bootstrap_differences(window, seed)
    again = bootstrap_differences(window, seed)
    reproducible = result.equals(again)
    return {"result": result, "reproducible": reproducible}


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda la tabla de diferencias por perfil."""
    out = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad" / "s4_metodo1_bootstrap.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results["result"].to_csv(out, index=False, encoding="utf-8")
    return [out]


# ===== 4. Gráficas =====
def chart_intervals(result: pd.DataFrame) -> go.Figure:
    """Puntos con barras de error: diferencia e intervalo por perfil estimable, ordenados."""
    r = result[result["estimable"]].sort_values("diferencia").reset_index(drop=True)
    colors = [CATEGORICAL[1] if c else "#b5b4ae" for c in r["concluyente"]]
    fig = go.Figure(go.Scatter(
        x=r.index, y=r["diferencia"], mode="markers", name="Diferencia",
        marker={"size": 8, "color": colors},
        error_y={"type": "data", "symmetric": False, "array": r["ic_superior"] - r["diferencia"],
                 "arrayminus": r["diferencia"] - r["ic_inferior"], "color": "#b5b4ae", "thickness": 1},
        customdata=r[["PERFIL", "ic_inferior", "ic_superior"]].values,
        hovertemplate="%{customdata[0]}<br>Diferencia %{y:+.2%}<br>IC [%{customdata[1]:+.2%}, %{customdata[2]:+.2%}]<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=TEXT_SECONDARY, line_width=1)
    fig.update_yaxes(tickformat="+.0%")
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(showlegend=False)
    return style(fig, f"S4 método 1: {CHALLENGER} − {CHAMPION} por perfil (naranja = intervalo excluye 0)",
                 "Perfiles ordenados por diferencia", "Diferencia de tasa de Over30 a 6 meses")


def make_charts(results: dict, config: dict) -> list:
    out = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad" / "graficas"
    return [save_html(chart_intervals(results["result"]), out / "s4_metodo1_intervalos.html")]


# ===== 5. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 1 si el bootstrap no es reproducible."""
    args = parse_args()
    config = load_config()
    set_seed(config["seed"])
    results = compute(config)
    written = save_results(results, config)
    if not args.sin_graficas:
        written += make_charts(results, config)

    r = results["result"]
    est, con = r["estimable"], r["concluyente"]
    print(f"S4 método 1: {len(r)} perfiles del soporte común; {N_BOOTSTRAP:,} remuestreos; intervalo {CI_LEVEL:.0%}")
    print(f"  Con intervalo estimable: {int(est.sum())}; sin intervalo (menos de 2 cohortes en algún lado): {int((~est).sum())}")
    print(f"  Concluyentes (intervalo excluye 0): {int(con.sum())} — DARIA con menos impago: {int((con & (r['diferencia'] < 0)).sum())}; "
          f"DARIA con más impago: {int((con & (r['diferencia'] > 0)).sum())}")
    w = r[est]
    print(f"  Ancho del intervalo (pp): mediana {100 * (w['ic_superior'] - w['ic_inferior']).median():.2f}, "
          f"rango {100 * (w['ic_superior'] - w['ic_inferior']).min():.2f}–{100 * (w['ic_superior'] - w['ic_inferior']).max():.2f}")
    print(f"  Reproducible (dos corridas idénticas): {'sí' if results['reproducible'] else 'NO'}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0 if results["reproducible"] else 1


if __name__ == "__main__":
    sys.exit(main())
