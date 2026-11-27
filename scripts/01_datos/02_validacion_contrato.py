"""02 — Validación del contrato de datos.

Objetivo:
    Aplicar las reglas bloqueantes y los avisos de CLAUDE.md §5 a la base estándar,
    comprobar con evidencia qué columnas son acumuladas dentro de una cohorte y a qué
    población se refieren REVENUE y EXPENSES_TOTAL.

Entradas:
    - data/processed/base_estandar.csv (de 01_carga_y_diccionario.py).
    - src/data/validation.py.

Salidas:
    - reports/01_datos/validacion.csv: una fila por regla (estado, filas afectadas,
      ejemplos de fila del CSV, qué falló y qué hacer).
    - reports/01_datos/columnas_acumuladas.csv: por columna, pasos de MOB que suben,
      bajan o se quedan igual dentro de la cohorte, y la conclusión.
    - reports/01_datos/poblacion_columnas.csv: comprobaciones sobre la población de
      REVENUE y EXPENSES_TOTAL.

Cómo correrlo (desde la raíz del repo):
    python scripts/01_datos/02_validacion_contrato.py
    python scripts/01_datos/02_validacion_contrato.py --entrada otro.csv --dir-reportes scratch/luis/val

Listo cuando:
    El validador pasa con el dataset real (código 0). Si alguna regla bloqueante
    falla, imprime cuál y sale con código 1.
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import catalog as cat  # noqa: E402
from src.data.load import load_standard  # noqa: E402
from src.data.validation import cumulative_evidence, has_blocking_failures, validate  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Valida la base estándar contra el contrato de datos (reglas bloqueantes y "
            "avisos), y documenta qué columnas son acumuladas y a qué población se refieren "
            "REVENUE y EXPENSES_TOTAL. Sale con código 1 si hay fallas bloqueantes."
        )
    )
    parser.add_argument("--sin-graficas", action="store_true", help="Este paso no genera gráficas; se acepta por uniformidad.")
    parser.add_argument("--entrada", type=Path, default=None,
                        help="CSV a validar, con nombres reales (por omisión, data/processed/base_estandar.csv).")
    parser.add_argument("--dir-reportes", type=Path, default=None,
                        help="Carpeta de salida de las tablas (por omisión, reports/01_datos/).")
    return parser.parse_args()


# ===== 2. Cálculo =====
def population_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Comprobaciones sobre la población de REVENUE y EXPENSES_TOTAL.

    Las descripciones del diccionario dicen que REVENUE y EXPENSES_TOTAL son "de las
    cuentas que se fueron a Over30". Si fuera así, en las filas sin cuentas en Over30
    (OVER30ACCTS = 0, que es acumulado) ambas valdrían 0.
    """
    no_o30 = df["OVER30ACCTS"] == 0
    total = df["EXPENSES_ACQ"] + df["EXPENSES_ECM_VAR"] + df["EXPENSES_ECM_FIX"]
    identity = np.isclose(df["EXPENSES_TOTAL"], total, rtol=1e-9, atol=1e-6)
    checks = [
        ("Filas con OVER30ACCTS = 0", int(no_o30.sum()), "Base de las dos comprobaciones siguientes."),
        ("…de ellas con REVENUE > 0 (%)", round(100 * (df.loc[no_o30, "REVENUE"] > 0).mean(), 2),
         "Si es > 0, REVENUE no se limita a las cuentas en Over30."),
        ("…de ellas con EXPENSES_TOTAL > 0 (%)", round(100 * (df.loc[no_o30, "EXPENSES_TOTAL"] > 0).mean(), 2),
         "Si es > 0, EXPENSES_TOTAL no se limita a las cuentas en Over30."),
        ("Filas donde EXPENSES_TOTAL = ACQ + VAR + FIX (%)", round(100 * identity.mean(), 4),
         "Si es 100, EXPENSES_TOTAL tiene la misma población que sus componentes."),
        ("Filas en MOB = 0 con REVENUE > 0 (%)", round(100 * (df.loc[df["MOB"] == 0, "REVENUE"] > 0).mean(), 2),
         "El ingreso empieza a acumularse después del booking."),
        ("Filas en MOB = 0 con OVER30ACCTS > 0 (%)", round(100 * (df.loc[df["MOB"] == 0, "OVER30ACCTS"] > 0).mean(), 2),
         "En el mes del booking no puede haber mora de más de 30 días."),
    ]
    return pd.DataFrame(checks, columns=["comprobacion", "valor", "interpretacion"])


def compute(input_path: Path) -> dict:
    """Carga la base estándar, valida y calcula la evidencia de columnas acumuladas."""
    df = load_standard(input_path)
    # Qué hace: cumulative_evidence() calcula Δ = x_MOB − x_(MOB−1) dentro de cada cohorte
    # para cada columna numérica. Por qué: una serie acumulada nunca baja al subir el MOB;
    # así se decide con datos cuáles columnas lo son (CLAUDE.md §5).
    return {
        "n_rows": len(df),
        "report": validate(df),
        "cumulative": cumulative_evidence(df, cat.VOLUME_COLUMNS + cat.RESULT_COLUMNS),
        "population": population_checks(df),
    }


# ===== 3. Guardado =====
def save_results(results: dict, reports: Path) -> list:
    """Guarda las tres tablas en la carpeta de reportes y devuelve las rutas."""
    reports.mkdir(parents=True, exist_ok=True)
    written = []
    for key, name in (("report", "validacion.csv"), ("cumulative", "columnas_acumuladas.csv"), ("population", "poblacion_columnas.csv")):
        out = reports / name
        results[key].to_csv(out, index=False, encoding="utf-8")
        written.append(out)
    return written


# ===== 4. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 0 si pasa y 1 si alguna regla bloqueante falla."""
    args = parse_args()
    config = load_config()
    set_seed(config["seed"])
    input_path = args.entrada or get_path(config, "processed_dir") / "base_estandar.csv"
    reports = args.dir_reportes or get_path(config, "reports_dir") / "01_datos"

    results = compute(input_path)
    report = results["report"]
    print(f"Validación del contrato sobre {results['n_rows']:,} filas:")
    for _, row in report.iterrows():
        extra = f"  ({row['filas_afectadas']:,} filas; ej. {row['ejemplo_filas']}) {row['que_fallo']}" if row["filas_afectadas"] else ""
        print(f"  [{row['estado']:5s}] {row['severidad']:10s} {row['regla']}{extra}")

    print("Columnas acumuladas (dentro de la cohorte, al subir el MOB):")
    for _, row in results["cumulative"].iterrows():
        flag = "  ← se esperaba acumulada" if row["se_esperaba_acumulada"] and row["conclusion"] != "acumulada" else ""
        print(f"  {row['columna']:18s} baja {row['pct_bajan']:6.2f}%  sube {row['pct_suben']:6.2f}%  → {row['conclusion']}{flag}")

    print("Población de REVENUE y EXPENSES_TOTAL:")
    for _, row in results["population"].iterrows():
        print(f"  {row['comprobacion']}: {row['valor']:g}")

    written = save_results(results, reports)
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")

    if has_blocking_failures(report):
        print("FALLA: hay reglas bloqueantes que no pasan; revisa reports/01_datos/validacion.csv.")
        return 1
    n_warn = int(((report["severidad"] == "aviso") & (report["filas_afectadas"] > 0)).sum())
    print(f"OK: todas las reglas bloqueantes pasan ({n_warn} avisos).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
