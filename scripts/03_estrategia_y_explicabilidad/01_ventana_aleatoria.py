"""01 — Ventana de asignación aleatoria para S4.

Objetivo:
    Tomar las cohortes maduras a H = 6 de la ventana aleatoria (202511–202607; maduras
    hasta 202603) de r.Reca champ y DARIA 1.6, armar la tabla por perfil
    (PRODUCT × CHANNEL × AGE_TAG), marcar el soporte común (perfiles con cuentas expuestas en
    las dos estrategias; los demás salen como "sin evidencia") y verificar el balance de
    perfiles entre estrategias antes y después de restringir al soporte común.

Entradas:
    - data/processed/s3_dataset.csv (de scripts/01_datos/04_madurez_y_curvas.py).
    - src/strategy/s4.py y src/strategy/balance.py.

Salidas:
    - data/processed/s4_ventana.csv: una fila por cohorte con PERFIL y SOPORTE_COMUN.
    - reports/03_estrategia_y_explicabilidad/s4_perfiles.csv: por perfil, cohortes, cuentas,
      eventos y tasa de cada estrategia, y su estado (soporte común o sin evidencia).
    - reports/03_estrategia_y_explicabilidad/s4_balance.csv: SMD por edad, producto y canal,
      en toda la ventana y en el soporte común.
    - reports/03_estrategia_y_explicabilidad/graficas/s4_balance.html

Cómo correrlo (desde la raíz del repo):
    python scripts/03_estrategia_y_explicabilidad/01_ventana_aleatoria.py

Listo cuando:
    El balance de perfiles entre estrategias queda verificado y documentado.
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from src.strategy.balance import SMD_THRESHOLD, categorical_balance  # noqa: E402
from src.strategy.s4 import CHALLENGER, CHAMPION, PROFILE_COLUMNS, STATUS_COMMON, WINDOW, build_window, profile_table  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402
from src.utils.charts import CATEGORICAL, TEXT_SECONDARY, save_html, style  # noqa: E402


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Arma la ventana aleatoria de S4 (cohortes maduras de r.Reca champ y DARIA 1.6), la "
            "tabla por perfil con su soporte común y el balance de perfiles entre estrategias."
        )
    )
    parser.add_argument("--sin-graficas", action="store_true", help="No genera la gráfica HTML.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def balance(window: pd.DataFrame, scope: str) -> pd.DataFrame:
    """SMD de cada categoría de edad, producto y canal entre champion y retadora (peso ACCOUNTS)."""
    parts = [categorical_balance(window, col, "STRATEGY", CHAMPION, CHALLENGER) for col in PROFILE_COLUMNS]
    return pd.concat(parts, ignore_index=True).assign(alcance=scope)


def compute(config: dict) -> dict:
    """Arma la ventana, la tabla por perfil y el balance antes y después del soporte común."""
    s3 = pd.read_csv(get_path(config, "processed_dir") / "s3_dataset.csv", encoding="utf-8")
    window = build_window(s3)
    profiles = profile_table(window)
    # Qué hace: compara la mezcla de perfiles (participación de cuentas por categoría) entre
    # estrategias en toda la ventana y solo en el soporte común. Por qué: con asignación
    # aleatoria las mezclas deberían parecerse; fuera del soporte común no hay contra qué comparar.
    bal = pd.concat([balance(window, "ventana completa"), balance(window[window["SOPORTE_COMUN"]], "soporte común")], ignore_index=True)
    return {"window": window, "profiles": profiles, "balance": bal}


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda la ventana en data/processed/ y las tablas en reports/03_estrategia_y_explicabilidad/."""
    reports = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad"
    reports.mkdir(parents=True, exist_ok=True)
    outputs = [
        (results["window"], get_path(config, "processed_dir") / "s4_ventana.csv"),
        (results["profiles"], reports / "s4_perfiles.csv"),
        (results["balance"], reports / "s4_balance.csv"),
    ]
    for table, path in outputs:
        table.to_csv(path, index=False, encoding="utf-8")
    return [p for _, p in outputs]


# ===== 4. Gráficas =====
def chart_balance(bal: pd.DataFrame) -> go.Figure:
    """Puntos: SMD por categoría, en toda la ventana y en el soporte común."""
    fig = go.Figure()
    order = bal[bal["alcance"] == "ventana completa"]
    labels = (order["variable"] + ": " + order["categoria"]).tolist()[::-1]
    for i, scope in enumerate(["ventana completa", "soporte común"]):
        b = bal[bal["alcance"] == scope].set_index(bal[bal["alcance"] == scope]["variable"] + ": " + bal[bal["alcance"] == scope]["categoria"])
        b = b.reindex(labels)
        fig.add_scatter(x=b["smd"], y=labels, mode="markers", name=scope.capitalize(), marker={"size": 9, "color": CATEGORICAL[i]},
                        hovertemplate="%{y}<br>SMD %{x:.3f}<extra>" + scope + "</extra>")
    for x in (-SMD_THRESHOLD, SMD_THRESHOLD):
        fig.add_vline(x=x, line_dash="dot", line_color=TEXT_SECONDARY)
    fig.update_layout(height=max(500, 22 * len(labels)))
    return style(fig, f"S4: balance de perfiles {CHAMPION} vs. {CHALLENGER}", f"SMD (positiva = más en {CHAMPION})", "")


def make_charts(results: dict, config: dict) -> list:
    """Guarda la gráfica de balance en reports/03_estrategia_y_explicabilidad/graficas/."""
    out = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad" / "graficas"
    return [save_html(chart_balance(results["balance"]), out / "s4_balance.html")]


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

    w, p, b = results["window"], results["profiles"], results["balance"]
    print(f"Ventana aleatoria {WINDOW[0]}–{WINDOW[1]}, cohortes maduras a 6 meses: vintages {sorted(w['VINTAGE'].unique().tolist())}")
    for s in (CHAMPION, CHALLENGER):
        ws = w[w["STRATEGY"] == s]
        print(f"  {s:13s} {len(ws):>5,} cohortes  {int(ws['ENRACCTS'].sum()):>7,} cuentas expuestas")
    common = p["estado"] == STATUS_COMMON
    print(f"Perfiles (producto × canal × edad): {len(p)}; en soporte común: {int(common.sum())}; sin evidencia: {int((~common).sum())}")
    print(f"  Cuentas expuestas fuera del soporte común: champ {int(p.loc[~common, 'cuentas_expuestas_champ'].sum()):,}, "
          f"DARIA {int(p.loc[~common, 'cuentas_expuestas_daria'].sum()):,}")
    for scope in ("ventana completa", "soporte común"):
        s = b[b["alcance"] == scope]
        worst = s.loc[s["smd"].abs().idxmax()]
        print(f"Balance ({scope}): {int(s['desbalance'].sum())} de {len(s)} categorías con |SMD| > {SMD_THRESHOLD}; "
              f"máxima {worst['variable']}: {worst['categoria']} ({worst['smd']:+.3f})")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
