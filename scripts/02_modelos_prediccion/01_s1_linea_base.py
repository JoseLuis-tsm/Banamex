"""01 — Líneas base de S1 (línea promedio en MOB = 0).

Objetivo:
    Medir dos líneas base contra las que se compara el XGBoost de S1:
      1. Simple: línea media ponderada por producto y score (Σ CREDITLINESAMTS / Σ ACCOUNTS
         del grupo en entrenamiento; si el grupo no se vio, solo producto y luego global).
      2. GLM Gamma (enlace log, scikit-learn) sobre las mismas variables del modelo, con
         peso = ACCOUNTS; la regularización alpha se elige en validación.

Entradas:
    - data/processed/s1_modelado.csv (de 00_variables_y_particion.py).

Salidas:
    - reports/02_modelos_prediccion/s1_linea_base_metricas.csv: MAE, MAPE, R² y sesgo
      ponderados en validación y prueba, total y por estrategia, para las dos líneas base.
    - reports/02_modelos_prediccion/s1_linea_base_glm_alpha.csv: MAE de validación de cada alpha.
    - models/s1/<versión>/modelo.json y metadata.json: el GLM Gamma como **modelo de reserva**
      (no se publica; actual.json no cambia), reajustado con todas las vintages y el alpha
      elegido. Las métricas de su metadata son las del GLM ajustado solo con entrenamiento.

Cómo correrlo (desde la raíz del repo):
    python scripts/02_modelos_prediccion/01_s1_linea_base.py

Listo cuando:
    Las métricas de ambas líneas base quedan guardadas y el GLM queda guardado como reserva.
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
from src.models import registry  # noqa: E402
from src.models.baselines import RatioBaseline  # noqa: E402
from src.models.glm import LinearGLM, save_reserve  # noqa: E402
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
def fit_glm(rows: pd.DataFrame, alpha: float) -> GammaRegressor:
    """GLM Gamma con enlace log: log E[línea] = β₀ + Σ βⱼ·xⱼ, ajustado con peso ACCOUNTS sobre `rows`."""
    model = GammaRegressor(alpha=alpha, max_iter=5000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # avisos de convergencia con alpha = 0; se revisa con el MAE
        model.fit(rows[feature_columns()], rows["LINEA_PROMEDIO"], sample_weight=rows["ACCOUNTS"])
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
        pred = fit_glm(train, alpha).predict(frame.loc[val, feature_columns()])
        alpha_rows.append({"alpha": alpha, "mae_validacion": regression_metrics(frame.loc[val, "LINEA_PROMEDIO"], pred, frame.loc[val, "ACCOUNTS"])["mae"]})
    alphas = pd.DataFrame(alpha_rows)
    best_alpha = float(alphas.loc[alphas["mae_validacion"].idxmin(), "alpha"])
    frame["pred_glm"] = fit_glm(train, best_alpha).predict(frame[feature_columns()])

    # Qué hace: reajusta el GLM con todas las vintages (las 33) y el alpha ya elegido, como los
    # modelos publicados. Por qué: es el modelo de reserva; la prueba ya se evaluó con el de arriba.
    reserve = LinearGLM.from_sklearn(fit_glm(frame, best_alpha), feature_columns(), "log")

    sets = [VALIDATION, TEST]
    cols = {"y": "LINEA_PROMEDIO", "w": "ACCOUNTS"}
    metrics = pd.concat([
        metrics_table(frame, SIMPLE_NAME, regression_metrics, sets, ["STRATEGY"], pred="pred_simple", **cols),
        metrics_table(frame, GLM_NAME, regression_metrics, sets, ["STRATEGY"], pred="pred_glm", **cols),
    ], ignore_index=True)
    metrics.insert(1, "detalle", np.where(metrics["modelo"] == GLM_NAME, f"alpha={best_alpha:g}", ""))
    return {
        "metrics": metrics, "alphas": alphas, "best_alpha": best_alpha, "reserve": reserve, "frame": frame,
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
    version, model_files = save_glm_reserve(results, config)
    results["reserve_version"] = version
    return [p for _, p in outputs] + model_files


def save_glm_reserve(results: dict, config: dict):
    """Guarda el GLM Gamma como reserva de S1 en models/s1/<versión>/ (sin publicarlo)."""
    frame, m = results["frame"], results["metrics"]
    glm = m[m["modelo"] == GLM_NAME]
    keys = ["mae", "mape_pct", "r2", "sesgo", "sesgo_pct", "peso", "filas"]

    def block(split: str, by: str) -> dict:
        rows = glm[(glm["conjunto"] == split) & (glm["desglose"] == by)]
        return {r["grupo"]: {k: float(r[k]) for k in keys} for _, r in rows.iterrows()}

    metadata = {
        "descripcion": "S1 (reserva): línea promedio esperada por cuenta en MOB = 0. GLM Gamma con enlace log sobre las variables del modelo, peso ACCOUNTS.",
        "objetivo": "LINEA_PROMEDIO = CREDITLINESAMTS / ACCOUNTS en MOB = 0",
        "formula": "línea = exp(β₀ + Σ βⱼ·xⱼ)",
        "peso": "ACCOUNTS",
        "variables": feature_columns(),
        "hiperparametros": {"alpha": results["best_alpha"], "elegido_en": "validacion (MAE ponderado)"},
        "vintages": {
            "entrenamiento": registry.vintage_list(frame, TRAIN), "validacion": registry.vintage_list(frame, VALIDATION),
            "prueba": registry.vintage_list(frame, TEST), "modelo_final": registry.vintage_list(frame),
        },
        "metricas": {
            "validacion": block(VALIDATION, "Total").get("Total"),
            "prueba": block(TEST, "Total").get("Total"),
            "prueba_por_estrategia": block(TEST, "STRATEGY"),
            "nota": "Métricas del GLM ajustado solo con entrenamiento. La prueba se evaluó una sola vez.",
        },
        "datos_sha256": registry.file_sha256(get_path(config, "raw_data")),
        "semilla": config["seed"],
        "base_sintetica": True,
    }
    return save_reserve(get_path(config, "models_dir"), "s1", results["reserve"], metadata)


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
    print(f"  GLM Gamma guardado como reserva (no publicado): models/s1/{results['reserve_version']}/")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
