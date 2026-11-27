"""06 — SHAP global.

Objetivo:
    Calcular los valores SHAP del modelo publicado de S3 para todas las cohortes de su
    conjunto de modelado, sumar las columnas one-hot por variable original y graficar la
    importancia global (media de |SHAP| ponderada por ENRACCTS).

    S3: TreeSHAP exacto de XGBoost (pred_contribs), en escala logit del modelo sin calibrar;
    se lee como dirección y tamaño relativo, no como puntos porcentuales.
    S1: pendiente de decisión (el S1 publicado es la línea base simple, no un árbol).

Entradas:
    - models/s3/actual.json y el modelo publicado.
    - data/processed/s3_modelado.csv.
    - src/explain/shap_values.py y src/explain/plots.py.

Salidas:
    - data/processed/shap_s3.csv: por cohorte, sus dimensiones, CONJUNTO, ENRACCTS, la
      contribución SHAP de cada variable original, VALOR_BASE y SALIDA_LOGIT.
    - reports/03_estrategia_y_explicabilidad/shap_s3_importancia_global.csv
    - reports/03_estrategia_y_explicabilidad/graficas/shap_s3_global_barras.html y
      shap_s3_global_beeswarm.html

Cómo correrlo (desde la raíz del repo):
    python scripts/03_estrategia_y_explicabilidad/06_shap_global.py

Listo cuando:
    Las barras globales de S3 están generadas (las de S1 se agregan al definir su explicación).
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

import pandas as pd  # noqa: E402

from src.data import catalog as cat  # noqa: E402
from src.explain.plots import beeswarm, importance_bars  # noqa: E402
from src.explain.shap_values import VARIABLES, additivity_error, global_importance, shap_by_variable, to_table, variable_values  # noqa: E402
from src.features.engineering import feature_columns  # noqa: E402
from src.models.predict import load_s3  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402
from src.utils.charts import save_html  # noqa: E402

ID_COLUMNS = cat.COHORT_COLUMNS + ["CONJUNTO", "ENRACCTS"]


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Calcula los valores SHAP de S3 por variable original y genera la importancia global (barras y beeswarm)."
    )
    parser.add_argument("--sin-graficas", action="store_true", help="No genera las gráficas HTML.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def compute(config: dict) -> dict:
    """Calcula SHAP agrupado para todas las cohortes de S3 y la importancia global."""
    s3 = load_s3(get_path(config, "models_dir"))
    frame = pd.read_csv(get_path(config, "processed_dir") / "s3_modelado.csv", encoding="utf-8")
    # Qué hace: φ por columna con TreeSHAP; suma las columnas one-hot de cada variable.
    # Por qué: al usuario le interesa "el producto", no "la columna PRODUCT__Costco".
    grouped = shap_by_variable(s3.booster, frame[feature_columns()])
    table = to_table(frame[ID_COLUMNS], grouped)  # contribuciones como SHAP_<variable>
    importance = global_importance(grouped, frame["ENRACCTS"].to_numpy())
    return {"table": table, "grouped": grouped, "features": frame[feature_columns()], "values": variable_values(frame),
            "importance": importance, "error": additivity_error(grouped), "version": s3.metadata["version"]}


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda los valores SHAP en data/processed/ y la importancia en reports/."""
    reports = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad"
    reports.mkdir(parents=True, exist_ok=True)
    outputs = [
        (results["table"], get_path(config, "processed_dir") / "shap_s3.csv"),
        (results["importance"], reports / "shap_s3_importancia_global.csv"),
    ]
    for table, path in outputs:
        table.to_csv(path, index=False, encoding="utf-8")
    return [p for _, p in outputs]


# ===== 4. Gráficas =====
def make_charts(results: dict, config: dict) -> list:
    """Barras de importancia y beeswarm global de S3."""
    out = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad" / "graficas"
    order = results["importance"]["variable"].tolist()
    return [
        save_html(importance_bars(results["importance"], "S3: importancia global de variables (SHAP)"), out / "shap_s3_global_barras.html"),
        save_html(beeswarm(results["grouped"], results["features"], results["values"], "S3: contribuciones SHAP por cohorte", order),
                  out / "shap_s3_global_beeswarm.html"),
    ]


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

    print(f"SHAP global de S3 (modelo {results['version']}, {len(results['table']):,} cohortes):")
    for _, row in results["importance"].iterrows():
        print(f"  {row['etiqueta']:22s} {row['importancia']:.4f}")
    print(f"  Aditividad (máximo |base + Σφ − salida logit|): {results['error']:.2e}")
    print(f"  Variables: {', '.join(VARIABLES)}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
