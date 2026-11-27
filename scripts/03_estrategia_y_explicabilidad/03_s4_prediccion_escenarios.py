"""03 — S4, método 2: diferencia predicha con S3 sobre las cohortes reales de cada estrategia.

Objetivo:
    Para cada perfil del soporte común (PRODUCT × CHANNEL × AGE_TAG), predecir con S3 la tasa
    de Over30 a 6 meses de las cohortes reales de cada estrategia, promediarla ponderando por
    ENRACCTS y calcular la diferencia DARIA 1.6 − r.Reca champ, con intervalo bootstrap sobre
    cohortes (2,000 remuestreos, semilla 42).

    Las predicciones se hacen dejando fuera una vintage a la vez: para cada una de las 5
    vintages de la ventana se reentrena S3 sin ella, con los hiperparámetros y rondas del
    modelo publicado (sin búsqueda nueva) y su misma calibración de Platt. Así ninguna cohorte
    se predice con un modelo que la vio, y la concordancia con el método 1 no sale inflada.
    Como referencia se guarda también la diferencia con el S3 publicado (en muestra).

    El intervalo refleja solo la variación entre cohortes, no la incertidumbre del modelo; la
    decisión usa el intervalo del método 1 (04_s4_sugerencia_y_validacion.py).

Entradas:
    - data/processed/s4_ventana.csv (de 01) y data/processed/s3_modelado.csv (Fase 2).
    - models/s3/actual.json y el modelo publicado (hiperparámetros, rondas, calibración).

Salidas:
    - data/processed/s4_ventana_predicha.csv: cohortes de la ventana con PRED_LOVO y
      PRED_PUBLICADO (tasa calibrada) y sus eventos esperados.
    - reports/03_estrategia_y_explicabilidad/s4_metodo2_prediccion.csv: por perfil, tasas
      predichas, diferencia, intervalo y diferencia en muestra (referencia).
    - reports/03_estrategia_y_explicabilidad/graficas/s4_metodo2_vs_metodo1.html

Cómo correrlo (desde la raíz del repo):
    python scripts/03_estrategia_y_explicabilidad/03_s4_prediccion_escenarios.py

Listo cuando:
    Cada perfil del soporte común tiene su diferencia predicha bajo ambas estrategias.
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from src.data import catalog as cat  # noqa: E402
from src.models.predict import load_s3  # noqa: E402
from src.strategy.s4 import CHALLENGER, CHAMPION, bootstrap_differences, leave_one_vintage_out  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402
from src.utils.charts import CATEGORICAL, TEXT_SECONDARY, save_html, style  # noqa: E402


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "S4 método 2: tasa predicha por S3 (dejando fuera una vintage a la vez) de las cohortes reales "
            "de cada estrategia, ponderada por ENRACCTS; diferencia DARIA 1.6 − r.Reca champ por perfil."
        )
    )
    parser.add_argument("--sin-graficas", action="store_true", help="No genera la gráfica HTML.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def compute(config: dict) -> dict:
    """Predicciones fuera de muestra por vintage, diferencias por perfil y referencia en muestra."""
    processed = get_path(config, "processed_dir")
    window = pd.read_csv(processed / "s4_ventana.csv", encoding="utf-8")
    frame = pd.read_csv(processed / "s3_modelado.csv", encoding="utf-8")
    s3 = load_s3(get_path(config, "models_dir"))
    meta = s3.metadata

    # Qué hace: 5 reentrenamientos, uno sin cada vintage de la ventana, con los parámetros
    # fijados en la Fase 2. Por qué: el publicado vio estas cohortes (ver docstring).
    window["PRED_LOVO"] = leave_one_vintage_out(frame, window, meta["hiperparametros"], int(meta["rondas"]), s3.a, s3.b)
    window["PRED_PUBLICADO"] = s3.predict_rate(window[cat.COHORT_COLUMNS])
    # Eventos esperados: E_i = n_i · p̂_i (n_i = ENRACCTS); Σ E_i / Σ n_i es la tasa predicha
    # del perfil ponderada por ENRACCTS.
    window["EVENTOS_LOVO"] = window["ENRACCTS"] * window["PRED_LOVO"]
    window["EVENTOS_PUBLICADO"] = window["ENRACCTS"] * window["PRED_PUBLICADO"]

    seed = config["seed"]
    lovo = bootstrap_differences(window, seed, events_column="EVENTOS_LOVO")
    insample = bootstrap_differences(window, seed, events_column="EVENTOS_PUBLICADO")
    result = lovo.merge(insample[["PERFIL", "diferencia"]].rename(columns={"diferencia": "diferencia_en_muestra"}), on="PERFIL")
    m1 = pd.read_csv(get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad" / "s4_metodo1_bootstrap.csv", encoding="utf-8")
    return {"window": window, "result": result, "m1": m1, "version": meta["version"],
            "obs_pred": float(window["OVER30ACCTS"].sum() / window["EVENTOS_LOVO"].sum())}


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda las cohortes con predicción y la tabla por perfil."""
    out = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad" / "s4_metodo2_prediccion.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results["result"].to_csv(out, index=False, encoding="utf-8")
    pred = get_path(config, "processed_dir") / "s4_ventana_predicha.csv"
    results["window"].to_csv(pred, index=False, encoding="utf-8")
    return [pred, out]


# ===== 4. Gráficas =====
def chart_vs_m1(result: pd.DataFrame, m1: pd.DataFrame) -> go.Figure:
    """Dispersión: diferencia observada (método 1) contra predicha (método 2) por perfil."""
    m = result[["PERFIL", "diferencia"]].merge(m1[["PERFIL", "diferencia", "concluyente"]], on="PERFIL", suffixes=("_m2", "_m1"))
    fig = go.Figure()
    for flag, name, color in ((False, "Método 1 no concluyente", "#b5b4ae"), (True, "Método 1 concluyente", CATEGORICAL[1])):
        d = m[m["concluyente"] == flag]
        fig.add_trace(go.Scatter(
            x=d["diferencia_m1"], y=d["diferencia_m2"], mode="markers", name=name, marker={"size": 8, "color": color},
            customdata=d[["PERFIL"]].values,
            hovertemplate="%{customdata[0]}<br>Observada %{x:+.2%}<br>Predicha %{y:+.2%}<extra></extra>",
        ))
    fig.add_hline(y=0, line_color=TEXT_SECONDARY, line_width=1)
    fig.add_vline(x=0, line_color=TEXT_SECONDARY, line_width=1)
    fig.update_xaxes(tickformat="+.0%")
    fig.update_yaxes(tickformat="+.0%")
    return style(fig, f"S4: diferencia {CHALLENGER} − {CHAMPION}, observada contra predicha (fuera de muestra)",
                 "Método 1: diferencia observada", "Método 2: diferencia predicha por S3")


def make_charts(results: dict, config: dict) -> list:
    out = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad" / "graficas"
    return [save_html(chart_vs_m1(results["result"], results["m1"]), out / "s4_metodo2_vs_metodo1.html")]


# ===== 5. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 1 si algún perfil del soporte común queda sin diferencia."""
    args = parse_args()
    config = load_config()
    set_seed(config["seed"])
    results = compute(config)
    written = save_results(results, config)
    if not args.sin_graficas:
        written += make_charts(results, config)

    r, w = results["result"], results["window"]
    missing = int(r["diferencia"].isna().sum()) + int(w.loc[w["SOPORTE_COMUN"], "PRED_LOVO"].isna().sum())
    print(f"S4 método 2 (S3 {results['version']}, dejando fuera una vintage a la vez: {', '.join(str(v) for v in sorted(w['VINTAGE'].unique()))})")
    print(f"  Cohortes predichas: {int(w['PRED_LOVO'].notna().sum()):,} de {len(w):,}; obs/pred en la ventana {results['obs_pred']:.4f}")
    print(f"  Perfiles del soporte común con diferencia predicha: {int(r['diferencia'].notna().sum())} de {len(r)}")
    print(f"  Diferencia predicha > 0 (DARIA con más impago): {int((r['diferencia'] > 0).sum())}; < 0: {int((r['diferencia'] < 0).sum())}")
    print(f"  Diferencia mediana: fuera de muestra {100 * r['diferencia'].median():+.2f} pp; "
          f"en muestra (referencia) {100 * r['diferencia_en_muestra'].median():+.2f} pp")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
