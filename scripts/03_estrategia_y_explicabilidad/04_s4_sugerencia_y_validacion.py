"""04 — S4: estrategia sugerida y concordancia entre métodos.

Objetivo:
    Combinar los dos métodos con la regla de decisión de src/strategy/s4.py (decide): se
    sugiere una estrategia solo si el intervalo del método 1 excluye cero y el método 2 tiene
    el mismo signo; si no, "Sin diferencia concluyente". Los perfiles fuera del soporte común
    salen como "Sin evidencia" y DARIA 2.0 como "No evaluable en riesgo".

    Criterio (CLAUDE.md §7): el signo coincide en al menos 70% de los perfiles con diferencia
    concluyente en el método 1. Se mide con el método 2 fuera de muestra (el que decide) y,
    solo como referencia, con el S3 publicado en muestra.

Entradas:
    - reports/03_estrategia_y_explicabilidad/s4_metodo1_bootstrap.csv (de 02).
    - reports/03_estrategia_y_explicabilidad/s4_metodo2_prediccion.csv (de 03).
    - reports/03_estrategia_y_explicabilidad/s4_perfiles.csv (de 01).

Salidas:
    - reports/03_estrategia_y_explicabilidad/s4_sugerencia.csv: por perfil, diferencias de
      ambos métodos, intervalo, sugerencia y motivo (más la fila de DARIA 2.0).
    - reports/03_estrategia_y_explicabilidad/s4_concordancia.csv
    - reports/03_estrategia_y_explicabilidad/graficas/s4_sugerencia.html

Cómo correrlo (desde la raíz del repo):
    python scripts/03_estrategia_y_explicabilidad/04_s4_sugerencia_y_validacion.py

Listo cuando:
    El signo coincide en al menos 70% de los perfiles con diferencia concluyente (si no, sale
    con código 1).
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from src.data import catalog as cat  # noqa: E402
from src.strategy.s4 import CHALLENGER, CHAMPION, NO_CONCLUSIVE, NO_EVIDENCE, NOT_RISK_EVALUABLE, decide, sign_agreement  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402
from src.utils.charts import CATEGORICAL, save_html, style  # noqa: E402

MIN_AGREEMENT = 0.70


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="S4: aplica la regla de decisión (intervalo del método 1 excluye cero y mismo signo en ambos métodos) "
                    "y verifica la concordancia de signo (al menos 70%%)."
    )
    parser.add_argument("--sin-graficas", action="store_true", help="No genera la gráfica HTML.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def compute(config: dict) -> dict:
    """Sugerencia por perfil y concordancia fuera de muestra y en muestra."""
    reports = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad"
    m1 = pd.read_csv(reports / "s4_metodo1_bootstrap.csv", encoding="utf-8")
    m2 = pd.read_csv(reports / "s4_metodo2_prediccion.csv", encoding="utf-8")
    profiles = pd.read_csv(reports / "s4_perfiles.csv", encoding="utf-8")
    decision = decide(m1, m2, profiles)
    reference = decide(m1, m2[["PERFIL", "diferencia_en_muestra"]].rename(columns={"diferencia_en_muestra": "diferencia"}), profiles)
    not_evaluable = pd.DataFrame([{"PERFIL": f"{s} (todos los perfiles)", "sugerencia": NOT_RISK_EVALUABLE,
                                   "motivo": "solo tiene MOB 0 y 1: sin desempeño de riesgo a 6 meses"} for s in sorted(cat.NOT_RISK_EVALUABLE)])
    agreement = pd.DataFrame([
        {"metodo_2": "fuera de muestra (decide)", **sign_agreement(decision)},
        {"metodo_2": "S3 publicado en muestra (referencia)", **sign_agreement(reference)},
    ])
    agreement["umbral"] = MIN_AGREEMENT
    agreement["cumple"] = agreement["concordancia"] >= MIN_AGREEMENT
    return {"decision": pd.concat([decision, not_evaluable], ignore_index=True), "agreement": agreement}


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    reports = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad"
    outputs = [(results["decision"], reports / "s4_sugerencia.csv"), (results["agreement"], reports / "s4_concordancia.csv")]
    for table, path in outputs:
        table.to_csv(path, index=False, encoding="utf-8")
    return [p for _, p in outputs]


# ===== 4. Gráficas =====
def chart_suggestion(decision: pd.DataFrame) -> go.Figure:
    """Barras horizontales: cuántos perfiles hay en cada resultado de la regla."""
    order = [CHAMPION, CHALLENGER, NO_CONCLUSIVE, NO_EVIDENCE]
    counts = decision["sugerencia"].value_counts().reindex(order, fill_value=0)
    labels = [f"Sugerida: {CHAMPION}", f"Sugerida: {CHALLENGER}", NO_CONCLUSIVE, NO_EVIDENCE]
    fig = go.Figure(go.Bar(
        y=labels, x=counts.values, orientation="h", marker={"color": [CATEGORICAL[0], CATEGORICAL[1], "#b5b4ae", "#d6d5d0"]},
        text=counts.values, textposition="outside", hovertemplate="%{y}: %{x} perfiles<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return style(fig, "S4: resultado de la regla de decisión por perfil (producto × canal × edad)", "Perfiles", "")


def make_charts(results: dict, config: dict) -> list:
    out = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad" / "graficas"
    return [save_html(chart_suggestion(results["decision"]), out / "s4_sugerencia.html")]


# ===== 5. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 1 si la concordancia fuera de muestra no llega a 70%."""
    args = parse_args()
    config = load_config()
    set_seed(config["seed"])
    results = compute(config)
    written = save_results(results, config)
    if not args.sin_graficas:
        written += make_charts(results, config)

    d, a = results["decision"], results["agreement"]
    print("S4: sugerencia por perfil")
    for label, n in d["sugerencia"].value_counts().items():
        print(f"  {label:30s} {n:>4}")
    print("Motivos de 'Sin diferencia concluyente':")
    for label, n in d.loc[d["sugerencia"] == NO_CONCLUSIVE, "motivo"].value_counts().items():
        print(f"  {label:70s} {n:>4}")
    print(f"Concordancia de signo (perfiles concluyentes en el método 1; umbral {MIN_AGREEMENT:.0%}):")
    for _, row in a.iterrows():
        print(f"  {row['metodo_2']:40s} {row['coinciden']}/{row['perfiles_concluyentes']} = {row['concordancia']:.1%}  "
              f"{'cumple' if row['cumple'] else 'NO CUMPLE'}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0 if bool(a.iloc[0]["cumple"]) else 1


if __name__ == "__main__":
    sys.exit(main())
