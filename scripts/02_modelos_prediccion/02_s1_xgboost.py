"""02 — XGBoost de S1 (línea promedio esperada).

Objetivo:
    Entrenar XGBoost sobre el logaritmo de la línea con peso = ACCOUNTS, buscar
    hiperparámetros con Optuna (50 pruebas, parada temprana de 50 rondas, semilla 42) en
    validación, evaluarlo una sola vez en prueba contra la línea base simple y, si cumple el
    criterio, reentrenarlo con todas las vintages y publicarlo.

Entradas:
    - data/processed/s1_modelado.csv (de 00_variables_y_particion.py).
    - reports/02_modelos_prediccion/s1_linea_base_metricas.csv (de 01_s1_linea_base.py).
    - data/raw/base_cards_sintetica.csv (solo para el hash SHA-256 de la versión de datos).

Salidas:
    - models/s1/<versión>/modelo.json (modelo final), modelo_evaluacion.json (entrenado sin
      validación ni prueba, el que se evaluó) y metadata.json.
    - models/s1/actual.json, solo si cumple el criterio.
    - reports/02_modelos_prediccion/s1_xgboost_optuna.csv, s1_xgboost_metricas.csv,
      s1_comparacion.csv y graficas/s1_*.html.

Cómo correrlo (desde la raíz del repo):
    python scripts/02_modelos_prediccion/02_s1_xgboost.py
    python scripts/02_modelos_prediccion/02_s1_xgboost.py --pruebas 10   # búsqueda corta, para revisar

Listo cuando:
    El MAE de prueba es al menos 10% menor que el de la línea base simple. Si no, no se
    publica y el script sale con código 1.
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import xgboost as xgb  # noqa: E402

from src.data import catalog as cat  # noqa: E402
from src.features.engineering import TEST, TRAIN, VALIDATION, feature_columns  # noqa: E402
from src.models import registry  # noqa: E402
from src.models.metrics import metrics_table, regression_metrics  # noqa: E402
from src.models.predict import MODEL_FILE  # noqa: E402
from src.models.tuning import N_TRIALS, predict_best, train_booster, tune  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402
from src.utils.charts import CATEGORICAL, save_html, style  # noqa: E402

MODEL_NAME = "XGBoost"
SIMPLE_NAME = "Base simple (producto × score)"
REQUIRED_IMPROVEMENT = 0.10  # MAE de prueba al menos 10% menor que el de la base simple


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Entrena el XGBoost de S1 sobre log(línea) con peso ACCOUNTS, busca hiperparámetros "
            "con Optuna en validación, lo evalúa una vez en prueba contra la línea base simple y "
            "lo publica en models/s1/ si su MAE es al menos 10%% menor."
        )
    )
    parser.add_argument("--sin-graficas", action="store_true", help="No genera las gráficas HTML.")
    parser.add_argument("--pruebas", type=int, default=N_TRIALS, help=f"Pruebas de Optuna (por omisión {N_TRIALS}, la del plan).")
    return parser.parse_args()


# ===== 2. Cálculo =====
def dmatrix(part: pd.DataFrame) -> xgb.DMatrix:
    """Matriz de XGBoost: etiqueta = log(línea), peso = ACCOUNTS."""
    return xgb.DMatrix(
        part[feature_columns()], label=np.log(part["LINEA_PROMEDIO"]), weight=part["ACCOUNTS"],
        feature_names=feature_columns(),
    )


def duan_factor(log_y: np.ndarray, log_pred: np.ndarray, w: np.ndarray) -> float:
    """Factor de retransformación de Duan (smearing), ponderado.

    Fórmula: s = Σ w·exp(r) / Σ w, con r = log(y) − f(x) los residuos en escala log.
    exp(f(x)) estima la media geométrica de la línea; multiplicarla por s la lleva a la media.
    """
    return float(np.sum(w * np.exp(log_y - log_pred)) / np.sum(w))


def compute(config: dict, n_trials: int) -> dict:
    """Busca hiperparámetros, evalúa en prueba, compara con la base y reentrena con todo."""
    seed = config["seed"]
    frame = pd.read_csv(get_path(config, "processed_dir") / "s1_modelado.csv", encoding="utf-8")
    parts = {s: frame[frame["CONJUNTO"] == s] for s in (TRAIN, VALIDATION, TEST)}
    dtrain, dval = dmatrix(parts[TRAIN]), dmatrix(parts[VALIDATION])

    # Qué hace: objetivo reg:squarederror sobre log(línea) con peso ACCOUNTS; Optuna minimiza
    # el MAE ponderado de validación en pesos (exp de la predicción). Por qué: el criterio
    # de publicación es el MAE en la escala original, no el error en logaritmo.
    base_params = {"objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist", "seed": seed, "nthread": 4}
    val = parts[VALIDATION]

    def val_mae(booster: xgb.Booster) -> float:
        return regression_metrics(val["LINEA_PROMEDIO"], np.exp(predict_best(booster, dval)), val["ACCOUNTS"])["mae"]

    params, rounds, trials = tune(base_params, dtrain, dval, val_mae, seed, n_trials)

    # Modelo de evaluación: solo entrenamiento, con el número de rondas elegido en validación.
    booster = train_booster(params, dtrain, num_rounds=rounds)
    frame["log_pred"] = booster.predict(xgb.DMatrix(frame[feature_columns()], feature_names=feature_columns()))

    # Qué hace: compara en validación la línea sin corregir (media geométrica) contra la
    # corregida con el factor de Duan estimado en entrenamiento, y se queda con la de menor MAE.
    tr = frame["CONJUNTO"] == TRAIN
    factor = duan_factor(np.log(frame.loc[tr, "LINEA_PROMEDIO"]), frame.loc[tr, "log_pred"], frame.loc[tr, "ACCOUNTS"])
    is_val = frame["CONJUNTO"] == VALIDATION
    options = {"ninguna": 1.0, "duan": factor}
    val_mae_by_option = {
        k: regression_metrics(frame.loc[is_val, "LINEA_PROMEDIO"], np.exp(frame.loc[is_val, "log_pred"]) * f, frame.loc[is_val, "ACCOUNTS"])["mae"]
        for k, f in options.items()
    }
    method = min(val_mae_by_option, key=val_mae_by_option.get)
    frame["pred_xgb"] = np.exp(frame["log_pred"]) * options[method]

    # Única evaluación en prueba (también validación, como referencia).
    metrics = metrics_table(frame, MODEL_NAME, regression_metrics, [VALIDATION, TEST], ["STRATEGY"],
                            y="LINEA_PROMEDIO", pred="pred_xgb", w="ACCOUNTS")
    baseline = pd.read_csv(get_path(config, "reports_dir") / "02_modelos_prediccion" / "s1_linea_base_metricas.csv", encoding="utf-8")
    comparison = pd.concat([baseline.drop(columns="detalle"), metrics], ignore_index=True)

    def total(model: str, split: str) -> float:
        row = comparison[(comparison["modelo"] == model) & (comparison["conjunto"] == split) & (comparison["desglose"] == "Total")]
        return float(row["mae"].iloc[0])

    test_mae, base_mae = total(MODEL_NAME, TEST), total(SIMPLE_NAME, TEST)
    improvement = 1 - test_mae / base_mae
    approved = improvement >= REQUIRED_IMPROVEMENT

    # Modelo final: todas las vintages elegibles (las 33), mismos parámetros y rondas.
    final = train_booster(params, dmatrix(frame), num_rounds=rounds)
    final_log_pred = final.predict(xgb.DMatrix(frame[feature_columns()], feature_names=feature_columns()))
    final_factor = duan_factor(np.log(frame["LINEA_PROMEDIO"]), final_log_pred, frame["ACCOUNTS"]) if method == "duan" else 1.0

    importance = pd.Series(booster.get_score(importance_type="gain")).sort_values(ascending=False)
    return {
        "frame": frame, "params": params, "rounds": rounds, "trials": trials, "metrics": metrics,
        "comparison": comparison, "test_mae": test_mae, "base_mae": base_mae, "improvement": improvement,
        "approved": approved, "booster": booster, "final": final, "method": method, "factor": final_factor,
        "val_mae_by_option": val_mae_by_option, "importance": importance,
    }


# ===== 3. Guardado =====
def vintage_list(frame: pd.DataFrame, split: str = None) -> list:
    part = frame if split is None else frame[frame["CONJUNTO"] == split]
    return sorted(int(v) for v in part["VINTAGE"].unique())


def save_results(results: dict, config: dict) -> list:
    """Guarda reportes, modelos y metadata; publica solo si cumple el criterio."""
    reports = get_path(config, "reports_dir") / "02_modelos_prediccion"
    reports.mkdir(parents=True, exist_ok=True)
    written = []
    for table, name in ((results["trials"], "s1_xgboost_optuna.csv"), (results["metrics"], "s1_xgboost_metricas.csv"), (results["comparison"], "s1_comparacion.csv")):
        table.to_csv(reports / name, index=False, encoding="utf-8")
        written.append(reports / name)

    frame = results["frame"]
    data_hash = registry.file_sha256(get_path(config, "raw_data"))
    hyper = {k: v for k, v in results["params"].items() if k not in ("nthread",)}
    version = registry.version_id("s1", {"datos": data_hash, "variables": feature_columns(), "params": hyper, "rondas": results["rounds"]})
    models_dir = get_path(config, "models_dir")
    folder = registry.version_dir(models_dir, "s1", version)
    results["final"].save_model(str(folder / MODEL_FILE))
    results["booster"].save_model(str(folder / "modelo_evaluacion.json"))

    test_total = results["metrics"][(results["metrics"]["conjunto"] == TEST) & (results["metrics"]["desglose"] == "Total")].iloc[0]
    val_total = results["metrics"][(results["metrics"]["conjunto"] == VALIDATION) & (results["metrics"]["desglose"] == "Total")].iloc[0]
    metric_keys = ["mae", "mape_pct", "r2", "sesgo", "sesgo_pct", "peso", "filas"]
    by_strategy = results["metrics"][(results["metrics"]["conjunto"] == TEST) & (results["metrics"]["desglose"] == "STRATEGY")]
    metadata = {
        "descripcion": "S1: línea promedio esperada por cuenta en MOB = 0. XGBoost sobre log(CREDITLINESAMTS / ACCOUNTS) con peso ACCOUNTS.",
        "objetivo": "LINEA_PROMEDIO = CREDITLINESAMTS / ACCOUNTS en MOB = 0",
        "transformacion_objetivo": "log",
        "peso": "ACCOUNTS",
        "variables": feature_columns(),
        "vintages": {
            "entrenamiento": vintage_list(frame, TRAIN), "validacion": vintage_list(frame, VALIDATION),
            "prueba": vintage_list(frame, TEST), "modelo_final": vintage_list(frame),
        },
        "hiperparametros": hyper,
        "rondas": results["rounds"],
        "busqueda": {"metodo": "Optuna TPE", "pruebas": int(len(results["trials"])), "semilla": config["seed"],
                     "parada_temprana": 50, "metrica_validacion": "MAE ponderado en pesos"},
        "retransformacion": {
            "metodo": results["method"], "factor": results["factor"],
            "mae_validacion_por_opcion": results["val_mae_by_option"],
            "nota": "Se elige en validación entre exp(f(x)) y exp(f(x))·factor de Duan; el factor del modelo final se recalcula con todas las vintages.",
        },
        "metricas": {
            "validacion": {k: float(val_total[k]) for k in metric_keys},
            "prueba": {k: float(test_total[k]) for k in metric_keys},
            "prueba_por_estrategia": {r["grupo"]: {k: float(r[k]) for k in metric_keys} for _, r in by_strategy.iterrows()},
            "nota": "Métricas del modelo de evaluación (entrenado sin validación ni prueba). La prueba se evaluó una sola vez.",
        },
        "linea_base": {"modelo": SIMPLE_NAME, "mae_prueba": results["base_mae"]},
        "criterio": {
            "regla": "MAE de prueba al menos 10% menor que el de la línea base simple",
            "mejora": results["improvement"], "aprobado": bool(results["approved"]),
        },
        "datos_sha256": data_hash,
        "semilla": config["seed"],
        "base_sintetica": True,
        "archivos": {"modelo": MODEL_FILE, "modelo_evaluacion": "modelo_evaluacion.json"},
    }
    written.append(registry.save_metadata(models_dir, "s1", version, metadata))
    results["version"] = version
    if results["approved"]:
        written.append(registry.publish(models_dir, "s1", version))
    return written


# ===== 4. Gráficas =====
def chart_mae_by_strategy(comparison: pd.DataFrame) -> go.Figure:
    """Barras agrupadas: MAE de prueba por estrategia, por modelo."""
    t = comparison[(comparison["conjunto"] == TEST) & (comparison["desglose"] == "STRATEGY")]
    fig = go.Figure()
    for i, model in enumerate(t["modelo"].unique()):
        s = t[t["modelo"] == model].set_index("grupo").reindex([x for x in cat.STRATEGIES if x in set(t["grupo"])])
        fig.add_bar(x=s.index, y=s["mae"], name=model, marker_color=CATEGORICAL[i],
                    hovertemplate=model + "<br>%{x}: MAE %{y:,.0f}<extra></extra>")
    fig.update_layout(barmode="group")
    return style(fig, "S1: MAE de prueba por estrategia", "Estrategia", "MAE ponderado (pesos)")


def chart_observed_vs_predicted(frame: pd.DataFrame) -> go.Figure:
    """Línea observada contra predicha en prueba, agregada por deciles de la predicción."""
    t = frame[frame["CONJUNTO"] == TEST].copy()
    t["decil"] = pd.qcut(t["pred_xgb"].rank(method="first"), 10, labels=False) + 1
    g = t.groupby("decil").apply(lambda d: pd.Series({
        "predicha": np.average(d["pred_xgb"], weights=d["ACCOUNTS"]),
        "observada": np.average(d["LINEA_PROMEDIO"], weights=d["ACCOUNTS"]),
    }), include_groups=False).reset_index()
    fig = go.Figure()
    fig.add_scatter(x=g["predicha"], y=g["observada"], mode="lines+markers", name="Deciles", marker_size=9,
                    line_color=CATEGORICAL[0], hovertemplate="Predicha %{x:,.0f}<br>Observada %{y:,.0f}<extra></extra>")
    lim = [0, float(max(g["predicha"].max(), g["observada"].max()) * 1.05)]
    fig.add_scatter(x=lim, y=lim, mode="lines", name="Igualdad", line={"dash": "dot", "color": "#8a8984", "width": 1})
    return style(fig, "S1: línea observada vs. predicha en prueba (por deciles)", "Línea predicha (media ponderada)", "Línea observada (media ponderada)")


def chart_importance(importance: pd.Series) -> go.Figure:
    """Barras horizontales: importancia por ganancia (las 20 variables principales)."""
    top = importance.head(20).iloc[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=CATEGORICAL[0], name="Ganancia",
                           hovertemplate="%{y}: %{x:,.2f}<extra></extra>"))
    fig.update_layout(height=600, showlegend=False)
    return style(fig, "S1: importancia de variables (ganancia, modelo de evaluación)", "Ganancia media", "")


def make_charts(results: dict, config: dict) -> list:
    """Construye las gráficas con Plotly y las guarda en reports/02_modelos_prediccion/graficas/."""
    out = get_path(config, "reports_dir") / "02_modelos_prediccion" / "graficas"
    return [
        save_html(chart_mae_by_strategy(results["comparison"]), out / "s1_mae_por_estrategia.html"),
        save_html(chart_observed_vs_predicted(results["frame"]), out / "s1_observado_vs_predicho.html"),
        save_html(chart_importance(results["importance"]), out / "s1_importancia_variables.html"),
    ]


# ===== 5. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 1 si el modelo no cumple el criterio (no se publica)."""
    args = parse_args()
    config = load_config()
    set_seed(config["seed"])

    results = compute(config, args.pruebas)
    written = save_results(results, config)
    if not args.sin_graficas:
        written += make_charts(results, config)

    c = results["comparison"]
    print(f"S1 XGBoost: {len(results['trials'])} pruebas de Optuna; {results['rounds']} rondas; "
          f"profundidad {results['params']['max_depth']}; retransformación: {results['method']} (factor final {results['factor']:.4f})")
    for _, row in c[c["desglose"] == "Total"].iterrows():
        print(f"  {row['modelo']:32s} {row['conjunto']:11s} MAE {row['mae']:>9,.0f}  MAPE {row['mape_pct']:5.1f}%  R² {row['r2']:6.3f}  sesgo {row['sesgo_pct']:+5.1f}%")
    print("  Prueba por estrategia (XGBoost):")
    m = results["metrics"]
    for _, row in m[(m["conjunto"] == TEST) & (m["desglose"] == "STRATEGY")].iterrows():
        print(f"    {row['grupo']:22s} MAE {row['mae']:>9,.0f}  MAPE {row['mape_pct']:5.1f}%  sesgo {row['sesgo_pct']:+5.1f}%  ({int(row['peso']):,} cuentas)")
    print(f"Criterio: MAE de prueba {results['test_mae']:,.0f} vs. base simple {results['base_mae']:,.0f} → mejora {results['improvement']:.1%} (se exige ≥ 10%)")
    print(f"  {'APROBADO: publicado como ' + results['version'] if results['approved'] else 'NO APROBADO: la versión ' + results['version'] + ' se guardó pero no se publicó.'}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0 if results["approved"] else 1


if __name__ == "__main__":
    sys.exit(main())
