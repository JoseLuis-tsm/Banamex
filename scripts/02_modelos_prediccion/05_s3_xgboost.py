"""05 — XGBoost de S3 (tasa acumulada de Over30 a 6 meses, calibrada).

Objetivo:
    Entrenar XGBoost con etiqueta fraccional (OVER30ACCTS / ENRACCTS), peso = ENRACCTS y
    restricción monótona decreciente en la posición del score; buscar hiperparámetros con
    Optuna (50 pruebas, parada temprana de 50 rondas, semilla 42) en validación; calibrar
    con Platt **solo en validación**; evaluar una sola vez en prueba contra el GLM y, si
    cumple los criterios, reentrenar con todas las vintages maduras y publicar.

    Calibración elegida: Platt, p_cal = σ(a · logit(p) + b). Por qué: tiene dos parámetros
    (la validación tiene solo 3 vintages; la isotónica se sobreajustaría), es estrictamente
    creciente (conserva el orden de riesgo, el AUC y la monotonía en el score) y no genera
    escalones con tasas repetidas.

Entradas:
    - data/processed/s3_modelado.csv (de 00_variables_y_particion.py).
    - reports/02_modelos_prediccion/s3_linea_base_metricas.csv (de 04_s3_linea_base.py).
    - data/raw/base_cards_sintetica.csv (solo para el hash SHA-256).

Salidas:
    - models/s3/<versión>/modelo.json, modelo_evaluacion.json y metadata.json (con la
      calibración y los umbrales de riesgo); models/s3/actual.json solo si cumple.
    - reports/02_modelos_prediccion/s3_xgboost_optuna.csv, s3_xgboost_metricas.csv,
      s3_comparacion.csv, s3_calibracion_deciles.csv, s3_niveles_riesgo.csv y
      graficas/s3_*.html.

Cómo correrlo (desde la raíz del repo):
    python scripts/02_modelos_prediccion/05_s3_xgboost.py

Listo cuando:
    En prueba: no es peor que el GLM (log loss ≤ y AUC ≥ los del GLM), la razón
    observado/predicho está entre 0.95 y 1.05 y la tasa observada crece con el nivel de
    riesgo. Si no, no se publica y sale con código 1.
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
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.data import catalog as cat  # noqa: E402
from src.features.engineering import SCORE_FEATURE, TEST, TRAIN, VALIDATION, build_features, feature_columns  # noqa: E402
from src.models import registry  # noqa: E402
from src.models.metrics import (  # noqa: E402
    calibration_by_decile, is_increasing, metrics_table, rate_by_risk_level, rate_metrics, risk_thresholds,
    score_monotonic_violations,
)
from src.models.predict import MODEL_FILE, logit, platt_transform  # noqa: E402
from src.models.tuning import N_TRIALS, predict_best, train_booster, tune  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402
from src.utils.charts import CATEGORICAL, save_html, style  # noqa: E402

MODEL_NAME = "XGBoost calibrado"
GLM_NAME = "GLM binomial"
OBS_PRED_RANGE = (0.95, 1.05)


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Entrena el XGBoost de S3 (etiqueta fraccional, peso ENRACCTS, monótono decreciente "
            "en el score), lo calibra con Platt en validación, lo evalúa una vez en prueba contra "
            "el GLM y lo publica en models/s3/ si cumple los criterios."
        )
    )
    parser.add_argument("--sin-graficas", action="store_true", help="No genera las gráficas HTML.")
    parser.add_argument("--pruebas", type=int, default=N_TRIALS, help=f"Pruebas de Optuna (por omisión {N_TRIALS}, la del plan).")
    return parser.parse_args()


# ===== 2. Cálculo =====
def dmatrix(part: pd.DataFrame) -> xgb.DMatrix:
    """Matriz de XGBoost: etiqueta = TASA_OVER30 (fracción), peso = ENRACCTS.

    Con objetivo binary:logistic, la pérdida de cada cohorte es
    −n·[y·log p + (1 − y)·log(1 − p)] = −[e·log p + (n − e)·log(1 − p)],
    con n = ENRACCTS, e = OVER30ACCTS e y = e/n: la verosimilitud binomial de la cohorte.
    """
    return xgb.DMatrix(part[feature_columns()], label=part["TASA_OVER30"], weight=part["ENRACCTS"], feature_names=feature_columns())


def fit_platt(p: np.ndarray, events: np.ndarray, exposed: np.ndarray):
    """Ajusta Platt en validación: logit P(evento) = a · logit(p) + b, con registros expandidos."""
    z = logit(p).reshape(-1, 1)
    X = np.vstack([z, z])
    y = np.concatenate([np.ones(len(z)), np.zeros(len(z))])
    w = np.concatenate([events, exposed - events]).astype(float)
    keep = w > 0
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(X[keep], y[keep], sample_weight=w[keep])
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def monotonic_check(booster: xgb.Booster, frame: pd.DataFrame) -> int:
    """Violaciones de la monotonía: para cada perfil de prueba, recorre todos los rangos de
    score de su familia (lo demás igual) y cuenta subidas de la tasa predicha."""
    profiles = frame.loc[frame["CONJUNTO"] == TEST, cat.COHORT_COLUMNS].drop_duplicates(cat.COHORT_COLUMNS[1:6]).copy()
    profiles["perfil"] = np.arange(len(profiles))
    rows = []
    for family, scores in cat.SCORES_BY_FAMILY.items():
        fam = profiles[profiles["STRATEGY"].map(cat.STRATEGY_FAMILY) == family]
        for s in scores:
            rows.append(fam.assign(SCORE_TAG=s))
    grid = pd.concat(rows, ignore_index=True)
    feats = build_features(grid)
    grid["p"] = booster.predict(xgb.DMatrix(feats, feature_names=feature_columns()))
    grid[SCORE_FEATURE] = feats[SCORE_FEATURE].to_numpy()
    return score_monotonic_violations(grid, "p", SCORE_FEATURE, ["perfil"])


def compute(config: dict, n_trials: int) -> dict:
    """Busca hiperparámetros, calibra en validación, evalúa en prueba y reentrena con todo."""
    seed = config["seed"]
    frame = pd.read_csv(get_path(config, "processed_dir") / "s3_modelado.csv", encoding="utf-8")
    parts = {s: frame[frame["CONJUNTO"] == s] for s in (TRAIN, VALIDATION, TEST)}
    dtrain, dval = dmatrix(parts[TRAIN]), dmatrix(parts[VALIDATION])

    # Qué hace: restricción monótona −1 en SCORE_POSICION (0 en el resto): al subir el score,
    # con lo demás igual, la tasa no puede subir. Optuna minimiza el log loss ponderado de validación.
    constraints = tuple(-1 if c == SCORE_FEATURE else 0 for c in feature_columns())
    base_params = {
        "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
        "monotone_constraints": str(constraints).replace(" ", ""), "seed": seed, "nthread": 4,
    }
    val = parts[VALIDATION]

    def val_logloss(booster: xgb.Booster) -> float:
        return rate_metrics(val["OVER30ACCTS"], val["ENRACCTS"], predict_best(booster, dval))["log_loss"]

    params, rounds, trials = tune(base_params, dtrain, dval, val_logloss, seed, n_trials)

    # Modelo de evaluación (solo entrenamiento) y calibración de Platt (solo validación).
    booster = train_booster(params, dtrain, num_rounds=rounds)
    frame["pred_raw"] = booster.predict(xgb.DMatrix(frame[feature_columns()], feature_names=feature_columns()))
    is_val = frame["CONJUNTO"] == VALIDATION
    a, b = fit_platt(frame.loc[is_val, "pred_raw"].to_numpy(), frame.loc[is_val, "OVER30ACCTS"].to_numpy(), frame.loc[is_val, "ENRACCTS"].to_numpy())
    frame["pred_xgb"] = platt_transform(frame["pred_raw"], a, b)

    # Umbrales de riesgo: percentiles 50/80/95 de la tasa predicha en entrenamiento, ponderada por ENRACCTS.
    is_train = frame["CONJUNTO"] == TRAIN
    thresholds = risk_thresholds(frame.loc[is_train, "pred_xgb"], frame.loc[is_train, "ENRACCTS"])

    # Única evaluación en prueba.
    sets, by = [VALIDATION, TEST], ["STRATEGY", "AGE_TAG"]
    cols = {"events": "OVER30ACCTS", "exposed": "ENRACCTS"}
    metrics = pd.concat([
        metrics_table(frame, "XGBoost sin calibrar", rate_metrics, sets, by, pred="pred_raw", **cols),
        metrics_table(frame, MODEL_NAME, rate_metrics, sets, by, pred="pred_xgb", **cols),
    ], ignore_index=True)
    baseline = pd.read_csv(get_path(config, "reports_dir") / "02_modelos_prediccion" / "s3_linea_base_metricas.csv", encoding="utf-8")
    comparison = pd.concat([baseline, metrics], ignore_index=True)

    test = frame[frame["CONJUNTO"] == TEST]
    deciles = pd.concat([
        calibration_by_decile(frame.loc[frame["CONJUNTO"] == s, "OVER30ACCTS"], frame.loc[frame["CONJUNTO"] == s, "ENRACCTS"],
                              frame.loc[frame["CONJUNTO"] == s, "pred_xgb"]).assign(conjunto=s)
        for s in sets
    ], ignore_index=True)
    levels = rate_by_risk_level(test["OVER30ACCTS"], test["ENRACCTS"], test["pred_xgb"], thresholds)

    def total(model: str, key: str) -> float:
        row = comparison[(comparison["modelo"] == model) & (comparison["conjunto"] == TEST) & (comparison["desglose"] == "Total")]
        return float(row[key].iloc[0])

    checks = {
        "log_loss_no_peor_que_glm": total(MODEL_NAME, "log_loss") <= total(GLM_NAME, "log_loss"),
        "auc_no_peor_que_glm": total(MODEL_NAME, "auc") >= total(GLM_NAME, "auc"),
        "obs_pred_en_rango": OBS_PRED_RANGE[0] <= total(MODEL_NAME, "obs_pred") <= OBS_PRED_RANGE[1],
        "tasa_crece_con_nivel": is_increasing(levels["tasa_observada"].tolist()),
    }
    violations = monotonic_check(booster, frame)
    checks["monotonia_en_score"] = violations == 0
    approved = all(checks.values())

    # Modelo final: las 27 vintages maduras, mismos parámetros y rondas; misma calibración.
    final = train_booster(params, dmatrix(frame), num_rounds=rounds)
    final_pred = platt_transform(final.predict(xgb.DMatrix(frame[feature_columns()], feature_names=feature_columns())), a, b)
    final_obs_pred = float(frame["OVER30ACCTS"].sum() / np.sum(frame["ENRACCTS"] * final_pred))

    importance = pd.Series(booster.get_score(importance_type="gain")).sort_values(ascending=False)
    return {
        "frame": frame, "params": params, "rounds": rounds, "trials": trials, "metrics": metrics, "comparison": comparison,
        "deciles": deciles, "levels": levels, "thresholds": thresholds, "platt": (a, b), "checks": checks,
        "violations": violations, "approved": approved, "booster": booster, "final": final,
        "final_obs_pred": final_obs_pred, "importance": importance, "totals": {
            k: {m: total(m, k) for m in (MODEL_NAME, GLM_NAME)} for k in ("auc", "ks", "gini", "log_loss", "brier", "obs_pred")
        },
    }


# ===== 3. Guardado =====
def vintage_list(frame: pd.DataFrame, split: str = None) -> list:
    part = frame if split is None else frame[frame["CONJUNTO"] == split]
    return sorted(int(v) for v in part["VINTAGE"].unique())


def save_results(results: dict, config: dict) -> list:
    """Guarda reportes, modelos y metadata; publica solo si cumple los criterios."""
    reports = get_path(config, "reports_dir") / "02_modelos_prediccion"
    reports.mkdir(parents=True, exist_ok=True)
    written = []
    for key, name in (("trials", "s3_xgboost_optuna.csv"), ("metrics", "s3_xgboost_metricas.csv"), ("comparison", "s3_comparacion.csv"),
                      ("deciles", "s3_calibracion_deciles.csv"), ("levels", "s3_niveles_riesgo.csv")):
        results[key].to_csv(reports / name, index=False, encoding="utf-8")
        written.append(reports / name)

    frame = results["frame"]
    data_hash = registry.file_sha256(get_path(config, "raw_data"))
    hyper = {k: v for k, v in results["params"].items() if k != "nthread"}
    a, b = results["platt"]
    version = registry.version_id("s3", {"datos": data_hash, "variables": feature_columns(), "params": hyper, "rondas": results["rounds"]})
    models_dir = get_path(config, "models_dir")
    folder = registry.version_dir(models_dir, "s3", version)
    results["final"].save_model(str(folder / MODEL_FILE))
    results["booster"].save_model(str(folder / "modelo_evaluacion.json"))

    m = results["metrics"][results["metrics"]["modelo"] == MODEL_NAME]
    keys = ["auc", "ks", "gini", "log_loss", "brier", "tasa_observada", "tasa_predicha", "obs_pred", "peso", "filas"]

    def block(split: str, by: str) -> dict:
        rows = m[(m["conjunto"] == split) & (m["desglose"] == by)]
        return {r["grupo"]: {k: float(r[k]) for k in keys} for _, r in rows.iterrows()}

    metadata = {
        "descripcion": "S3: tasa acumulada de Over30 a 6 meses (OVER30ACCTS / ENRACCTS en MOB = 6), calibrada. XGBoost con etiqueta fraccional y peso ENRACCTS.",
        "objetivo": "TASA_OVER30 = OVER30ACCTS / ENRACCTS en MOB = 6 (OVER30ACCTS acumulado)",
        "peso": "ENRACCTS",
        "variables": feature_columns(),
        "restriccion_monotona": {"variable": SCORE_FEATURE, "sentido": "decreciente", "violaciones_en_prueba": results["violations"]},
        "vintages": {
            "entrenamiento": vintage_list(frame, TRAIN), "validacion": vintage_list(frame, VALIDATION),
            "prueba": vintage_list(frame, TEST), "modelo_final": vintage_list(frame),
        },
        "hiperparametros": hyper,
        "rondas": results["rounds"],
        "busqueda": {"metodo": "Optuna TPE", "pruebas": int(len(results["trials"])), "semilla": config["seed"],
                     "parada_temprana": 50, "metrica_validacion": "log loss ponderado"},
        "calibracion": {
            "metodo": "Platt", "formula": "p_cal = sigmoid(a * logit(p) + b)", "a": a, "b": b, "ajustada_en": "validacion",
            "por_que": "Dos parámetros (validación de 3 vintages), estrictamente creciente (conserva orden, AUC y monotonía) y sin escalones.",
            "obs_pred_modelo_final_en_muestra": results["final_obs_pred"],
        },
        "umbrales_riesgo": {
            **results["thresholds"],
            "definicion": "Percentiles 50, 80 y 95 de la tasa predicha calibrada en entrenamiento, ponderados por ENRACCTS. Bajo ≤ P50 < Medio ≤ P80 < Alto ≤ P95 < Crítico.",
        },
        "metricas": {
            "validacion": block(VALIDATION, "Total").get("Total"),
            "prueba": block(TEST, "Total").get("Total"),
            "prueba_por_estrategia": block(TEST, "STRATEGY"),
            "prueba_por_edad": block(TEST, "AGE_TAG"),
            "prueba_por_nivel_riesgo": results["levels"].to_dict(orient="records"),
            "nota": "Métricas del modelo de evaluación (entrenado solo con entrenamiento, calibrado en validación). La prueba se evaluó una sola vez.",
        },
        "linea_base": {"modelo": GLM_NAME, "prueba": {k: v[GLM_NAME] for k, v in results["totals"].items()}},
        "criterio": {
            "reglas": "En prueba: log loss ≤ y AUC ≥ los del GLM; obs/pred entre 0.95 y 1.05; tasa observada creciente por nivel de riesgo; sin violaciones de la monotonía en el score.",
            "resultado": {k: bool(v) for k, v in results["checks"].items()}, "aprobado": bool(results["approved"]),
        },
        "datos_sha256": data_hash,
        "semilla": config["seed"],
        "base_sintetica": True,
        "archivos": {"modelo": MODEL_FILE, "modelo_evaluacion": "modelo_evaluacion.json"},
    }
    written.append(registry.save_metadata(models_dir, "s3", version, metadata))
    results["version"] = version
    if results["approved"]:
        written.append(registry.publish(models_dir, "s3", version))
    return written


# ===== 4. Gráficas =====
def chart_calibration(deciles: pd.DataFrame) -> go.Figure:
    """Tasa observada contra predicha por decil, en validación y prueba."""
    fig = go.Figure()
    for i, split in enumerate([VALIDATION, TEST]):
        d = deciles[deciles["conjunto"] == split]
        fig.add_scatter(x=d["tasa_predicha"], y=d["tasa_observada"], mode="lines+markers", name=split.capitalize(),
                        marker_size=9, line_color=CATEGORICAL[i], customdata=d["decil"],
                        hovertemplate="Decil %{customdata}<br>Predicha %{x:.2%}<br>Observada %{y:.2%}<extra></extra>")
    lim = [0, float(max(deciles["tasa_predicha"].max(), deciles["tasa_observada"].max()) * 1.05)]
    fig.add_scatter(x=lim, y=lim, mode="lines", name="Igualdad", line={"dash": "dot", "color": "#8a8984", "width": 1})
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return style(fig, "S3: calibración por deciles (XGBoost calibrado)", "Tasa predicha", "Tasa observada")


def chart_metric_by_strategy(comparison: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """Barras agrupadas: una métrica de prueba por estrategia, por modelo."""
    t = comparison[(comparison["conjunto"] == TEST) & (comparison["desglose"] == "STRATEGY") & (comparison["modelo"] != "XGBoost sin calibrar")]
    fig = go.Figure()
    for i, model in enumerate(t["modelo"].unique()):
        s = t[t["modelo"] == model].set_index("grupo").reindex([x for x in cat.STRATEGIES if x in set(t["grupo"])])
        fig.add_bar(x=s.index, y=s[metric], name=model, marker_color=CATEGORICAL[i],
                    hovertemplate=model + "<br>%{x}: %{y:.4f}<extra></extra>")
    fig.update_layout(barmode="group")
    return style(fig, title, "Estrategia", metric)


def chart_levels(levels: pd.DataFrame) -> go.Figure:
    """Barras: tasa observada y predicha por nivel de riesgo en prueba."""
    fig = go.Figure()
    fig.add_bar(x=levels["nivel"], y=levels["tasa_observada"], name="Observada", marker_color=CATEGORICAL[0],
                hovertemplate="%{x}: %{y:.2%} observada<extra></extra>")
    fig.add_bar(x=levels["nivel"], y=levels["tasa_predicha"], name="Predicha", marker_color=CATEGORICAL[1],
                hovertemplate="%{x}: %{y:.2%} predicha<extra></extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%")
    return style(fig, "S3: tasa de Over30 por nivel de riesgo (prueba)", "Nivel de riesgo", "Tasa a 6 meses")


def make_charts(results: dict, config: dict) -> list:
    """Construye las gráficas con Plotly y las guarda en reports/02_modelos_prediccion/graficas/."""
    out = get_path(config, "reports_dir") / "02_modelos_prediccion" / "graficas"
    top = results["importance"].head(20).iloc[::-1]
    imp = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=CATEGORICAL[0], name="Ganancia",
                           hovertemplate="%{y}: %{x:,.2f}<extra></extra>"))
    imp.update_layout(height=600, showlegend=False)
    return [
        save_html(chart_calibration(results["deciles"]), out / "s3_calibracion_deciles.html"),
        save_html(chart_metric_by_strategy(results["comparison"], "auc", "S3: AUC de prueba por estrategia"), out / "s3_auc_por_estrategia.html"),
        save_html(chart_levels(results["levels"]), out / "s3_niveles_riesgo.html"),
        save_html(style(imp, "S3: importancia de variables (ganancia, modelo de evaluación)", "Ganancia media", ""), out / "s3_importancia_variables.html"),
    ]


# ===== 5. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 1 si el modelo no cumple los criterios (no se publica)."""
    args = parse_args()
    config = load_config()
    set_seed(config["seed"])

    results = compute(config, args.pruebas)
    written = save_results(results, config)
    if not args.sin_graficas:
        written += make_charts(results, config)

    a, b = results["platt"]
    print(f"S3 XGBoost: {len(results['trials'])} pruebas de Optuna; {results['rounds']} rondas; profundidad "
          f"{results['params']['max_depth']}; Platt a = {a:.4f}, b = {b:.4f}")
    c = results["comparison"]
    for _, row in c[c["desglose"] == "Total"].iterrows():
        print(f"  {row['modelo']:31s} {row['conjunto']:11s} AUC {row['auc']:.4f}  KS {row['ks']:.4f}  "
              f"log loss {row['log_loss']:.5f}  Brier {row['brier']:.5f}  obs/pred {row['obs_pred']:.3f}")
    m = results["metrics"]
    for by in ("STRATEGY", "AGE_TAG"):
        print(f"  Prueba por {by} ({MODEL_NAME}):")
        for _, row in m[(m["modelo"] == MODEL_NAME) & (m["conjunto"] == TEST) & (m["desglose"] == by)].iterrows():
            print(f"    {row['grupo']:22s} AUC {row['auc']:.4f}  obs/pred {row['obs_pred']:.3f}  ({int(row['peso']):,} cuentas expuestas)")
    t = results["thresholds"]
    print(f"  Umbrales de riesgo (entrenamiento): P50 {t['p50']:.4f}  P80 {t['p80']:.4f}  P95 {t['p95']:.4f}")
    for _, row in results["levels"].iterrows():
        print(f"    {row['nivel']:8s} observada {row['tasa_observada']:.4f}  predicha {row['tasa_predicha']:.4f}  ({int(row['cuentas_expuestas']):,} cuentas)")
    print("Criterios en prueba:")
    for k, v in results["checks"].items():
        print(f"  [{'ok' if v else 'NO'}] {k}")
    print(f"  {'APROBADO: publicado como ' + results['version'] if results['approved'] else 'NO APROBADO: la versión ' + results['version'] + ' se guardó pero no se publicó.'}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0 if results["approved"] else 1


if __name__ == "__main__":
    sys.exit(main())
