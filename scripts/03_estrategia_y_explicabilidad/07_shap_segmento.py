"""07 — SHAP por segmento.

Objetivo:
    Mostrar la importancia (barras) y la distribución de contribuciones (beeswarm) de S3
    para un subconjunto de cohortes elegido por parámetro: una variable y un valor (por
    ejemplo, CHANNEL = Internet).

Entradas:
    - data/processed/shap_s3.csv (de 06_shap_global.py).
    - src/explain/.

Salidas:
    - reports/03_estrategia_y_explicabilidad/shap_s3_segmento_<variable>_<valor>.csv
    - reports/03_estrategia_y_explicabilidad/graficas/shap_s3_segmento_<variable>_<valor>_barras.html
      y _beeswarm.html

Cómo correrlo (desde la raíz del repo):
    python scripts/03_estrategia_y_explicabilidad/07_shap_segmento.py
    python scripts/03_estrategia_y_explicabilidad/07_shap_segmento.py --variable PRODUCT --valor Costco
    python scripts/03_estrategia_y_explicabilidad/07_shap_segmento.py --variable STRATEGY --valor "DARIA 1.6"

Listo cuando:
    Se puede pedir cualquier segmento por parámetro. Si la variable o el valor no existen,
    lista los disponibles y sale con código 1.
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402
import re  # noqa: E402

import pandas as pd  # noqa: E402

from src.explain.plots import beeswarm, importance_bars  # noqa: E402
from src.explain.shap_values import VARIABLE_LABELS, from_table, global_importance, variable_values  # noqa: E402
from src.features.engineering import build_features  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402
from src.utils.charts import save_html  # noqa: E402

FILTER_VARIABLES = ["STRATEGY", "STRATEGY_SEGMENT", "PRODUCT", "CHANNEL", "AGE_TAG", "SCORE_TAG", "CONJUNTO"]


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="SHAP de S3 filtrado por un segmento: barras de importancia y beeswarm.")
    parser.add_argument("--variable", default="CHANNEL", help=f"Variable de filtro: {', '.join(FILTER_VARIABLES)} (por omisión CHANNEL).")
    parser.add_argument("--valor", default="Internet", help="Valor de la variable (por omisión Internet).")
    parser.add_argument("--sin-graficas", action="store_true", help="No genera las gráficas HTML.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def compute(config: dict, variable: str, value: str) -> dict:
    """Filtra las cohortes del segmento y calcula su importancia."""
    table = pd.read_csv(get_path(config, "processed_dir") / "shap_s3.csv", encoding="utf-8")
    if variable not in FILTER_VARIABLES:
        raise ValueError(f"Variable '{variable}' no válida. Usa una de: {', '.join(FILTER_VARIABLES)}.")
    available = sorted(table[variable].astype(str).unique())
    if value not in available:
        raise ValueError(f"El valor '{value}' no aparece en {variable}. Disponibles: {'; '.join(available)}.")
    seg = table[table[variable].astype(str) == value].copy()
    shap_seg = from_table(seg)
    importance = global_importance(shap_seg, seg["ENRACCTS"].to_numpy())
    overall = global_importance(from_table(table), table["ENRACCTS"].to_numpy())
    importance["importancia_global"] = importance["variable"].map(overall.set_index("variable")["importancia"])
    return {"segment": seg, "shap": shap_seg, "importance": importance, "features": build_features(seg), "values": variable_values(seg)}


# ===== 3. Guardado =====
def slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")


def save_results(results: dict, config: dict, variable: str, value: str) -> list:
    """Guarda la importancia del segmento (con la global para comparar)."""
    out = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad" / f"shap_s3_segmento_{slug(variable)}_{slug(value)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results["importance"].to_csv(out, index=False, encoding="utf-8")
    return [out]


# ===== 4. Gráficas =====
def make_charts(results: dict, config: dict, variable: str, value: str) -> list:
    """Barras y beeswarm del segmento."""
    out = get_path(config, "reports_dir") / "03_estrategia_y_explicabilidad" / "graficas"
    name = f"shap_s3_segmento_{slug(variable)}_{slug(value)}"
    label = f"{VARIABLE_LABELS.get(variable, variable)} = {value}"
    order = results["importance"]["variable"].tolist()
    seg = results["shap"]
    return [
        save_html(importance_bars(results["importance"], f"S3: importancia SHAP en {label}"), out / f"{name}_barras.html"),
        save_html(beeswarm(seg, results["features"], results["values"], f"S3: contribuciones SHAP en {label}", order), out / f"{name}_beeswarm.html"),
    ]


# ===== 5. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 1 si la variable o el valor no existen."""
    args = parse_args()
    config = load_config()
    set_seed(config["seed"])
    try:
        results = compute(config, args.variable, args.valor)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    written = save_results(results, config, args.variable, args.valor)
    if not args.sin_graficas:
        written += make_charts(results, config, args.variable, args.valor)

    seg = results["segment"]
    print(f"SHAP de S3 en {args.variable} = {args.valor}: {len(seg):,} cohortes, {int(seg['ENRACCTS'].sum()):,} cuentas expuestas")
    for _, row in results["importance"].iterrows():
        print(f"  {row['etiqueta']:22s} segmento {row['importancia']:.4f}   global {row['importancia_global']:.4f}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
