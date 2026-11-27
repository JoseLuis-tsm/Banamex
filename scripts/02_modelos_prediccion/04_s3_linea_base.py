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

Salidas (reports/02_modelos_prediccion/):
    - s3_linea_base_metricas.csv: AUC, KS, Gini, log loss, Brier y obs/pred ponderados en
      validación y prueba; total, por estrategia y por rango de edad.
    - s3_linea_base_calibracion.csv: calibración por deciles en validación y prueba.
    - s3_linea_base_glm_c.csv: log loss de validación de cada C probado.

Cómo correrlo (desde la raíz del repo):
    python scripts/02_modelos_prediccion/04_s3_linea_base.py

Listo cuando:
    Las métricas de ambas líneas base quedan guardadas.
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
from src.models.baselines import RatioBaseline  # noqa: E402
from src.models.metrics import calibration_by_decile, metrics_table, rate_metrics  # noqa: E402
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
    return {"metrics": metrics, "calibration": calibration, "cs": cs, "best_c": best_c}


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
    return [p for _, p in outputs]


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
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
