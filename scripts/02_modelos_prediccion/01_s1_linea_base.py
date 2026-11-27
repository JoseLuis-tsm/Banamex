"""01 — Líneas base de S1 (línea promedio en MOB = 0).

Objetivo:
    Medir dos líneas base contra las que se compara el XGBoost de S1:
      1. Simple: línea media ponderada por producto y score (Σ CREDITLINESAMTS / Σ ACCOUNTS
         del grupo en entrenamiento; si el grupo no se vio, solo producto y luego global).
      2. GLM Gamma (enlace log, scikit-learn) sobre las mismas variables del modelo, con
         peso = ACCOUNTS; la regularización alpha se elige en validación.

Entradas:
    - data/processed/s1_modelado.csv (de 00_variables_y_particion.py).

Salidas (reports/02_modelos_prediccion/):
    - s1_linea_base_metricas.csv: MAE, MAPE, R² y sesgo ponderados en validación y prueba,
      total y por estrategia, para las dos líneas base.
    - s1_linea_base_glm_alpha.csv: MAE de validación de cada alpha probado.

Cómo correrlo (desde la raíz del repo):
    python scripts/02_modelos_prediccion/01_s1_linea_base.py

Listo cuando:
    Las métricas de ambas líneas base quedan guardadas.
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import GammaRegressor  # noqa: E402

from src.features.engineering import TEST, TRAIN, VALIDATION, feature_columns  # noqa: E402
from src.models.baselines import RatioBaseline  # noqa: E402
from src.models.metrics import metrics_table, regression_metrics  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402

GLM_ALPHAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
SIMPLE_NAME = "Base simple (producto × score)"
GLM_NAME = "GLM Gamma"


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Calcula las líneas base de S1 (media ponderada por producto y score; GLM Gamma) "
            "y sus métricas ponderadas en validación y prueba."
        )
    )
    parser.add_argument("--sin-graficas", action="store_true", help="Este paso no genera gráficas; se acepta por uniformidad.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def fit_glm(frame: pd.DataFrame, alpha: float) -> GammaRegressor:
    """GLM Gamma con enlace log: log E[línea] = β₀ + Σ βⱼ·xⱼ, ajustado con peso ACCOUNTS en entrenamiento."""
    train = frame[frame["CONJUNTO"] == TRAIN]
    model = GammaRegressor(alpha=alpha, max_iter=5000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # avisos de convergencia con alpha = 0; se revisa con el MAE
        model.fit(train[feature_columns()], train["LINEA_PROMEDIO"], sample_weight=train["ACCOUNTS"])
    return model


def compute(config: dict) -> dict:
    """Ajusta las dos líneas base en entrenamiento y las evalúa en validación y prueba."""
    frame = pd.read_csv(get_path(config, "processed_dir") / "s1_modelado.csv", encoding="utf-8")
    train = frame[frame["CONJUNTO"] == TRAIN]

    # Qué hace: RatioBaseline guarda Σ CREDITLINESAMTS / Σ ACCOUNTS por (producto, score),
    # por producto y global, calculado solo con entrenamiento. Por qué: la media de la línea
    # ponderada por cuentas es exactamente esa razón de sumas.
    simple = RatioBaseline([["PRODUCT", "SCORE_TAG"], ["PRODUCT"], []], "CREDITLINESAMTS", "ACCOUNTS").fit(train)
    frame["pred_simple"] = simple.predict(frame)
    fallback = pd.Series(simple.last_levels).map({0: "producto × score", 1: "solo producto", 2: "global"})

    # Qué hace: prueba cada alpha en entrenamiento y elige el de menor MAE ponderado en
    # validación. Por qué: la prueba no se usa para decidir nada.
    val = frame["CONJUNTO"] == VALIDATION
    alpha_rows = []
    for alpha in GLM_ALPHAS:
        pred = fit_glm(frame, alpha).predict(frame.loc[val, feature_columns()])
        alpha_rows.append({"alpha": alpha, "mae_validacion": regression_metrics(frame.loc[val, "LINEA_PROMEDIO"], pred, frame.loc[val, "ACCOUNTS"])["mae"]})
    alphas = pd.DataFrame(alpha_rows)
    best_alpha = float(alphas.loc[alphas["mae_validacion"].idxmin(), "alpha"])
    frame["pred_glm"] = fit_glm(frame, best_alpha).predict(frame[feature_columns()])

    sets = [VALIDATION, TEST]
    cols = {"y": "LINEA_PROMEDIO", "w": "ACCOUNTS"}
    metrics = pd.concat([
        metrics_table(frame, SIMPLE_NAME, regression_metrics, sets, ["STRATEGY"], pred="pred_simple", **cols),
        metrics_table(frame, GLM_NAME, regression_metrics, sets, ["STRATEGY"], pred="pred_glm", **cols),
    ], ignore_index=True)
    metrics.insert(1, "detalle", np.where(metrics["modelo"] == GLM_NAME, f"alpha={best_alpha:g}", ""))
    return {
        "metrics": metrics, "alphas": alphas, "best_alpha": best_alpha,
        "fallback": fallback.groupby(frame["CONJUNTO"]).value_counts().rename("filas").reset_index(),
    }


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda las métricas y la búsqueda de alpha en reports/02_modelos_prediccion/."""
    reports = get_path(config, "reports_dir") / "02_modelos_prediccion"
    reports.mkdir(parents=True, exist_ok=True)
    outputs = [
        (results["metrics"], reports / "s1_linea_base_metricas.csv"),
        (results["alphas"], reports / "s1_linea_base_glm_alpha.csv"),
        (results["fallback"], reports / "s1_linea_base_respaldo.csv"),
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
    print(f"Líneas base de S1 (GLM Gamma con alpha = {results['best_alpha']:g}, elegido en validación):")
    for _, row in m[m["desglose"] == "Total"].iterrows():
        print(f"  {row['modelo']:32s} {row['conjunto']:11s} MAE {row['mae']:>9,.0f}  MAPE {row['mape_pct']:5.1f}%  "
              f"R² {row['r2']:6.3f}  sesgo {row['sesgo_pct']:+5.1f}%")
    print("  Nivel usado por la base simple (filas):")
    for _, row in results["fallback"].iterrows():
        print(f"    {row['CONJUNTO']:13s} {row.iloc[1]:18s} {row['filas']:,}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
