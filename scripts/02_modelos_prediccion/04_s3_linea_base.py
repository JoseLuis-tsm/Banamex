"""04 — Líneas base de S3 (tasa acumulada de Over30 a 6 meses).

Objetivo:
    Medir dos líneas base contra las que se compara el XGBoost de S3:
      1. Simple: tasa ponderada por score y producto (Σ OVER30ACCTS / Σ ENRACCTS del grupo en
         entrenamiento; si el grupo no se vio, solo producto y luego global).
      2. GLM binomial (regresión logística de scikit-learn) sobre las mismas variables del
         modelo: cada cohorte aporta OVER30ACCTS eventos y ENRACCTS − OVER30ACCTS no
         eventos. La regularización C se elige en validación por log loss.

Entradas:
    - data/processed/s3_modelado.csv (de 00_variables_y_particion.py).

Salidas:
    - reports/02_modelos_prediccion/s3_linea_base_metricas.csv: AUC, KS, Gini, log loss,
      Brier y obs/pred ponderados en validación y prueba; total, por estrategia y por edad.
    - reports/02_modelos_prediccion/s3_linea_base_calibracion.csv: calibración por deciles.
    - reports/02_modelos_prediccion/s3_linea_base_glm_c.csv: log loss de validación de cada C.
    - models/s3/<versión>/modelo.json y metadata.json: el GLM binomial como **modelo de
      reserva** (no se publica; actual.json no cambia), reajustado con las 27 vintages maduras
      y el C elegido, con sus propios umbrales de riesgo. Las métricas de su metadata son las
      del GLM ajustado solo con entrenamiento.

Cómo correrlo (desde la raíz del repo):
    python scripts/02_modelos_prediccion/04_s3_linea_base.py

Listo cuando:
    Las métricas de ambas líneas base quedan guardadas y el GLM queda guardado como reserva.
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.features.engineering import TEST, TRAIN, VALIDATION, feature_columns  # noqa: E402
from src.models import registry  # noqa: E402
from src.models.baselines import RatioBaseline  # noqa: E402
from src.models.glm import LinearGLM, save_reserve  # noqa: E402
from src.models.metrics import calibration_by_decile, metrics_table, rate_metrics, risk_thresholds  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402

GLM_CS = [0.01, 0.1, 1.0, 10.0, 100.0]
SIMPLE_NAME = "Base simple (score × producto)"
GLM_NAME = "GLM binomial"


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Calcula las líneas base de S3 (tasa ponderada por score y producto; GLM binomial) "
            "y sus métricas ponderadas y calibración en validación y prueba."
        )
    )
    parser.add_argument("--sin-graficas", action="store_true", help="Este paso no genera gráficas; se acepta por uniformidad.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def expand_binomial(part: pd.DataFrame):
    """Duplica cada cohorte: (x, y=1, peso=OVER30ACCTS) y (x, y=0, peso=ENRACCTS − OVER30ACCTS)."""
    x = part[feature_columns()].to_numpy()
    events = part["OVER30ACCTS"].to_numpy(dtype=float)
    non_events = part["ENRACCTS"].to_numpy(dtype=float) - events
    X = np.vstack([x, x])
    y = np.concatenate([np.ones(len(part)), np.zeros(len(part))])
    w = np.concatenate([events, non_events])
    keep = w > 0
    return X[keep], y[keep], w[keep]


def fit_glm(train: pd.DataFrame, c: float) -> LogisticRegression:
    """GLM binomial: logit P(Over30) = β₀ + Σ βⱼ·xⱼ, ajustado en registros expandidos."""
    X, y, w = expand_binomial(train)
    return LogisticRegression(C=c, max_iter=5000).fit(X, y, sample_weight=w)


def compute(config: dict) -> dict:
    """Ajusta las dos líneas base en entrenamiento y las evalúa en validación y prueba."""
    frame = pd.read_csv(get_path(config, "processed_dir") / "s3_modelado.csv", encoding="utf-8")
    train = frame[frame["CONJUNTO"] == TRAIN]
    val = frame[frame["CONJUNTO"] == VALIDATION]

    # Qué hace: Σ OVER30ACCTS / Σ ENRACCTS por (score, producto), por producto y global, solo
    # con entrenamiento. Por qué: es la tasa ponderada por cuentas expuestas del grupo.
    simple = RatioBaseline([["SCORE_TAG", "PRODUCT"], ["PRODUCT"], []], "OVER30ACCTS", "ENRACCTS").fit(train)
    frame["pred_simple"] = simple.predict(frame)

    c_rows = []
    for c in GLM_CS:
        p = fit_glm(train, c).predict_proba(val[feature_columns()].to_numpy())[:, 1]
        c_rows.append({"C": c, "log_loss_validacion": rate_metrics(val["OVER30ACCTS"], val["ENRACCTS"], p)["log_loss"]})
    cs = pd.DataFrame(c_rows)
    best_c = float(cs.loc[cs["log_loss_validacion"].idxmin(), "C"])
    frame["pred_glm"] = fit_glm(train, best_c).predict_proba(frame[feature_columns()].to_numpy())[:, 1]

    # Umbrales de la reserva: percentiles 50/80/95 de su tasa predicha en entrenamiento,
    # ponderada por ENRACCTS (la misma definición que el S3 publicado, CLAUDE.md §7).
    is_train = frame["CONJUNTO"] == TRAIN
    thresholds = risk_thresholds(frame.loc[is_train, "pred_glm"], frame.loc[is_train, "ENRACCTS"])
    # Qué hace: reajusta el GLM con las 27 vintages maduras y el C ya elegido, como el S3
    # publicado. Por qué: es el modelo de reserva; la prueba ya se evaluó con el de arriba.
    reserve = LinearGLM.from_sklearn(fit_glm(frame, best_c), feature_columns(), "logit")

    sets, by = [VALIDATION, TEST], ["STRATEGY", "AGE_TAG"]
    cols = {"events": "OVER30ACCTS", "exposed": "ENRACCTS"}
    metrics = pd.concat([
        metrics_table(frame, SIMPLE_NAME, rate_metrics, sets, by, pred="pred_simple", **cols),
        metrics_table(frame, GLM_NAME, rate_metrics, sets, by, pred="pred_glm", **cols),
    ], ignore_index=True)

    calibration = []
    for model, col in ((SIMPLE_NAME, "pred_simple"), (GLM_NAME, "pred_glm")):
        for split in sets:
            part = frame[frame["CONJUNTO"] == split]
            calibration.append(calibration_by_decile(part["OVER30ACCTS"], part["ENRACCTS"], part[col]).assign(modelo=model, conjunto=split))
    calibration = pd.concat(calibration, ignore_index=True)
    return {"metrics": metrics, "calibration": calibration, "cs": cs, "best_c": best_c,
            "reserve": reserve, "thresholds": thresholds, "frame": frame}


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda métricas, calibración y búsqueda de C en reports/02_modelos_prediccion/."""
    reports = get_path(config, "reports_dir") / "02_modelos_prediccion"
    reports.mkdir(parents=True, exist_ok=True)
    outputs = [
        (results["metrics"], reports / "s3_linea_base_metricas.csv"),
        (results["calibration"], reports / "s3_linea_base_calibracion.csv"),
        (results["cs"], reports / "s3_linea_base_glm_c.csv"),
    ]
    for table, path in outputs:
        table.to_csv(path, index=False, encoding="utf-8")
    version, model_files = save_glm_reserve(results, config)
    results["reserve_version"] = version
    return [p for _, p in outputs] + model_files


def save_glm_reserve(results: dict, config: dict):
    """Guarda el GLM binomial como reserva de S3 en models/s3/<versión>/ (sin publicarlo)."""
    frame, m = results["frame"], results["metrics"]
    glm = m[m["modelo"] == GLM_NAME]
    keys = ["auc", "gini", "ks", "log_loss", "brier", "tasa_observada", "tasa_predicha", "obs_pred", "peso", "filas"]

    def block(split: str, by: str) -> dict:
        rows = glm[(glm["conjunto"] == split) & (glm["desglose"] == by)]
        return {r["grupo"]: {k: float(r[k]) for k in keys} for _, r in rows.iterrows()}

    metadata = {
        "descripcion": "S3 (reserva): tasa acumulada de Over30 a 6 meses (OVER30ACCTS / ENRACCTS en MOB = 6). GLM binomial (regresión logística) sobre las variables del modelo; cada cohorte aporta OVER30ACCTS eventos y ENRACCTS − OVER30ACCTS no eventos.",
        "objetivo": "TASA_OVER30 = OVER30ACCTS / ENRACCTS en MOB = 6 (OVER30ACCTS acumulado)",
        "formula": "tasa = σ(β₀ + Σ βⱼ·xⱼ); sin calibración aparte",
        "peso": "ENRACCTS",
        "variables": feature_columns(),
        "hiperparametros": {"C": results["best_c"], "elegido_en": "validacion (log loss ponderado)"},
        "vintages": {
            "entrenamiento": registry.vintage_list(frame, TRAIN), "validacion": registry.vintage_list(frame, VALIDATION),
            "prueba": registry.vintage_list(frame, TEST), "modelo_final": registry.vintage_list(frame),
        },
        "umbrales_riesgo": {
            **results["thresholds"],
            "definicion": "Percentiles 50, 80 y 95 de la tasa predicha por el GLM en entrenamiento, ponderados por ENRACCTS. Bajo ≤ P50 < Medio ≤ P80 < Alto ≤ P95 < Crítico.",
        },
        "metricas": {
            "validacion": block(VALIDATION, "Total").get("Total"),
            "prueba": block(TEST, "Total").get("Total"),
            "prueba_por_estrategia": block(TEST, "STRATEGY"),
            "prueba_por_edad": block(TEST, "AGE_TAG"),
            "nota": "Métricas del GLM ajustado solo con entrenamiento. La prueba se evaluó una sola vez.",
        },
        "datos_sha256": registry.file_sha256(get_path(config, "raw_data")),
        "semilla": config["seed"],
        "base_sintetica": True,
    }
    return save_reserve(get_path(config, "models_dir"), "s3", results["reserve"], metadata)


# ===== 4. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 0 si todo salió bien."""
    parse_args()
    config = load_config()
    set_seed(config["seed"])

    results = compute(config)
    written = save_results(results, config)

    m = results["metrics"]
    print(f"Líneas base de S3 (GLM binomial con C = {results['best_c']:g}, elegido en validación):")
    for _, row in m[m["desglose"] == "Total"].iterrows():
        print(f"  {row['modelo']:31s} {row['conjunto']:11s} AUC {row['auc']:.4f}  KS {row['ks']:.4f}  "
              f"log loss {row['log_loss']:.5f}  Brier {row['brier']:.5f}  obs/pred {row['obs_pred']:.3f}")
    print(f"  GLM binomial guardado como reserva (no publicado): models/s3/{results['reserve_version']}/")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
