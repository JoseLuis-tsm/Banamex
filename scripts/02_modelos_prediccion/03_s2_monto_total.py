"""03 — S2: monto total esperado (cuentas planeadas × S1).

Objetivo:
    Aplicar la función determinista S2 = cuentas planeadas × S1 con el modelo de S1
    publicado y comprobar que el resultado es exactamente ese producto. Como ejemplo de
    uso, toma como "cuentas planeadas" las cuentas reales de cada cohorte de prueba y
    compara el monto esperado con el monto de línea observado, total y por estrategia.

Entradas:
    - models/s1/actual.json y el modelo publicado (de 02_s1_xgboost.py).
    - data/processed/s1_modelado.csv (cohortes de prueba).
    - src/models/s2.py.

Salidas:
    - reports/02_modelos_prediccion/s2_monto_total.csv: por estrategia, cuentas, monto
      esperado (S2), monto observado (Σ CREDITLINESAMTS) y diferencia relativa.

Cómo correrlo (desde la raíz del repo):
    python scripts/02_modelos_prediccion/03_s2_monto_total.py

Listo cuando:
    S2 coincide exactamente con cuentas planeadas × S1 en todas las filas. Si no hay un
    modelo de S1 publicado, avisa y sale con código 1.
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.engineering import TEST  # noqa: E402
from src.models.predict import load_s1  # noqa: E402
from src.models.s2 import expected_total_amount  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Calcula S2 = cuentas planeadas × S1 con el modelo de S1 publicado y verifica que sea exacto."
    )
    parser.add_argument("--sin-graficas", action="store_true", help="Este paso no genera gráficas; se acepta por uniformidad.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def compute(config: dict) -> dict:
    """Predice S1 por cohorte de prueba, aplica S2 y resume por estrategia."""
    s1 = load_s1(get_path(config, "models_dir"))
    frame = pd.read_csv(get_path(config, "processed_dir") / "s1_modelado.csv", encoding="utf-8")
    test = frame[frame["CONJUNTO"] == TEST].copy()

    # Qué hace: S1 = línea esperada por cuenta; S2 = ACCOUNTS (como cuentas planeadas) × S1.
    test["S1"] = s1.predict_line(test)
    test["S2"] = expected_total_amount(test["ACCOUNTS"].to_numpy(), test["S1"].to_numpy())
    exact = bool(np.array_equal(test["S2"].to_numpy(), test["ACCOUNTS"].to_numpy(dtype=float) * test["S1"].to_numpy()))

    rows = []
    for name, g in [("Total", test)] + list(test.groupby("STRATEGY")):
        rows.append({
            "estrategia": name, "cohortes": len(g), "cuentas": int(g["ACCOUNTS"].sum()),
            "monto_esperado_s2": float(g["S2"].sum()), "monto_observado": float(g["CREDITLINESAMTS"].sum()),
        })
    summary = pd.DataFrame(rows)
    summary["diferencia_pct"] = 100 * (summary["monto_esperado_s2"] / summary["monto_observado"] - 1)
    return {"summary": summary, "exact": exact, "version": s1.metadata["version"]}


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda el resumen en reports/02_modelos_prediccion/s2_monto_total.csv."""
    out = get_path(config, "reports_dir") / "02_modelos_prediccion" / "s2_monto_total.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results["summary"].to_csv(out, index=False, encoding="utf-8")
    return [out]


# ===== 4. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 1 si no hay S1 publicado o S2 no es exacto."""
    parse_args()
    config = load_config()
    set_seed(config["seed"])
    try:
        results = compute(config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc} Corre antes 02_s1_xgboost.py; S2 depende de un S1 publicado.")
        return 1
    written = save_results(results, config)

    print(f"S2 = cuentas planeadas × S1 (modelo {results['version']}):")
    print(f"  Exacto en todas las filas: {'sí' if results['exact'] else 'NO'}")
    for _, row in results["summary"].iterrows():
        print(f"  {row['estrategia']:22s} {row['cuentas']:>9,} cuentas  S2 {row['monto_esperado_s2']:>16,.0f}  "
              f"observado {row['monto_observado']:>16,.0f}  ({row['diferencia_pct']:+.1f}%)")
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")
    return 0 if results["exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
