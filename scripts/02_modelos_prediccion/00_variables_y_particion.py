"""00 — Variables de entrada y partición temporal.

Objetivo:
    Construir las variables de entrada de S1 y S3 (CLAUDE.md §7), asignar cada vintage a
    entrenamiento, validación o prueba y comprobar que no hay fuga de información.

Entradas:
    - data/processed/s1_dataset.csv y s3_dataset.csv (de scripts/01_datos/04_madurez_y_curvas.py).
    - configs/config.json: split (train, validation, test, test_s1).
    - src/features/engineering.py.

Salidas:
    - data/processed/s1_modelado.csv y s3_modelado.csv: dimensiones, pesos, objetivo,
      CONJUNTO y las variables de entrada.
    - reports/02_modelos_prediccion/particion.csv: vintages, filas y peso por conjunto.

Cómo correrlo (desde la raíz del repo):
    python scripts/02_modelos_prediccion/00_variables_y_particion.py

Listo cuando:
    La prueba de fuga pasa y el número de vintages por conjunto coincide con CLAUDE.md §7
    (20 / 3 / 4 para S3; 20 / 3 / 10 para S1). Si no, sale con código 1.
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

import pandas as pd  # noqa: E402

from src.features.engineering import SPLIT_COLUMN, TEST, TRAIN, VALIDATION, feature_columns, model_frame  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402

# Vintages esperadas por conjunto (CLAUDE.md §7).
EXPECTED_VINTAGES = {
    "S1": {TRAIN: 20, VALIDATION: 3, TEST: 10},
    "S3": {TRAIN: 20, VALIDATION: 3, TEST: 4},
}


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Construye las variables de entrada de S1 y S3, asigna la partición temporal por "
            "vintage y verifica que no haya fuga. Escribe s1_modelado.csv y s3_modelado.csv."
        )
    )
    parser.add_argument("--sin-graficas", action="store_true", help="Este paso no genera gráficas; se acepta por uniformidad.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def split_summary(frame: pd.DataFrame, output: str, weight: str) -> pd.DataFrame:
    """Vintages (primera, última, cuántas), filas y peso total por conjunto."""
    t = frame.groupby(SPLIT_COLUMN).agg(
        vintage_inicial=("VINTAGE", "min"), vintage_final=("VINTAGE", "max"),
        vintages=("VINTAGE", "nunique"), filas=("VINTAGE", "size"), peso=(weight, "sum"),
    ).reindex([TRAIN, VALIDATION, TEST]).reset_index()
    t.insert(0, "salida", output)
    t["peso_columna"] = weight
    t["vintages_esperadas"] = t[SPLIT_COLUMN].map(EXPECTED_VINTAGES[output])
    return t


def compute(config: dict) -> dict:
    """Arma las dos tablas de modelado y sus comprobaciones."""
    processed = get_path(config, "processed_dir")
    s1 = pd.read_csv(processed / "s1_dataset.csv", encoding="utf-8")
    s3 = pd.read_csv(processed / "s3_dataset.csv", encoding="utf-8")
    # Qué hace: model_frame() agrega CONJUNTO (por rango de vintage de config["split"]) y las
    # variables. Por qué: la partición es temporal por vintage, nunca aleatoria por fila;
    # S1 usa test_s1 (hasta 202609) porque la línea se observa desde MOB = 0.
    s1_frame, s1_problems = model_frame(s1, config["split"], "test_s1")
    s3_frame, s3_problems = model_frame(s3, config["split"], "test")
    summary = pd.concat([split_summary(s1_frame, "S1", "ACCOUNTS"), split_summary(s3_frame, "S3", "ENRACCTS")], ignore_index=True)
    return {
        "s1": s1_frame, "s3": s3_frame, "summary": summary,
        "problems": {"S1": s1_problems, "S3": s3_problems},
        "dropped": {"S1": len(s1) - len(s1_frame), "S3": len(s3) - len(s3_frame)},
    }


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda las tablas de modelado y el resumen de la partición."""
    processed = get_path(config, "processed_dir")
    reports = get_path(config, "reports_dir") / "02_modelos_prediccion"
    reports.mkdir(parents=True, exist_ok=True)
    outputs = [
        (results["s1"], processed / "s1_modelado.csv"),
        (results["s3"], processed / "s3_modelado.csv"),
        (results["summary"], reports / "particion.csv"),
    ]
    for table, path in outputs:
        table.to_csv(path, index=False, encoding="utf-8")
    return [p for _, p in outputs]


# ===== 4. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 1 si hay fuga o la partición no es la esperada."""
    parse_args()
    config = load_config()
    set_seed(config["seed"])

    results = compute(config)
    written = save_results(results, config)

    print(f"Variables de entrada: {len(feature_columns())}")
    ok = True
    for _, row in results["summary"].iterrows():
        match = row["vintages"] == row["vintages_esperadas"]
        ok &= bool(match)
        print(f"  {row['salida']} {row[SPLIT_COLUMN]:13s} {row['vintage_inicial']}–{row['vintage_final']}  "
              f"{row['vintages']:>2} vintages (esperadas {row['vintages_esperadas']}) {'ok' if match else 'NO COINCIDE'}  "
              f"{row['filas']:>6,} filas  peso {row['peso_columna']} {int(row['peso']):,}")
    for output, problems in results["problems"].items():
        print(f"Prueba de fuga {output}: {'pasa' if not problems else 'FALLA'}")
        for p in problems:
            print(f"  - {p}")
        ok &= not problems
        if results["dropped"][output]:
            print(f"  Filas fuera de todos los conjuntos (no se usan): {results['dropped'][output]:,}")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
