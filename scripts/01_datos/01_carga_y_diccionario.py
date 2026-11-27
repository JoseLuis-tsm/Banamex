"""01 — Carga del CSV y diccionario de variables.

Objetivo:
    Cargar la base cruda con los nombres de columna reales (acepta también los alias
    dummiN), normalizar los espacios de las categorías y comprobar que el catálogo de
    src/data/catalog.py coincide con el diccionario.

Entradas:
    - data/raw/base_cards_sintetica.csv (ruta raw_data de configs/config.json).
    - data/contracts/diccionario_variables.csv (ruta dictionary).
    - src/data/dictionary.py, src/data/load.py, src/data/catalog.py.

Salidas:
    - data/contracts/column_map.json: mapa alias ↔ nombre real y formato encontrado.
    - data/processed/base_estandar.csv: base con nombres reales y categorías sin espacios.
    - reports/01_datos/carga_perfil_columnas.csv: tipo, nulos, mínimo y máximo por columna.
    - reports/01_datos/carga_catalogo_vs_datos.csv: por dimensión, valores del catálogo
      presentes y ausentes en los datos, y valores de los datos fuera del catálogo.

Cómo correrlo (desde la raíz del repo):
    python scripts/01_datos/01_carga_y_diccionario.py

Listo cuando:
    El CSV se carga con nombres estándar, se imprime qué formato traía y el catálogo
    coincide con el diccionario (si no coincide, sale con código 1).
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402
import json  # noqa: E402

import pandas as pd  # noqa: E402

from src.data import catalog as cat  # noqa: E402
from src.data.dictionary import (  # noqa: E402
    catalog_differences, column_map, dictionary_catalog, read_dictionary, values_with_trailing_spaces,
)
from src.data.load import ColumnFormatError, load_raw  # noqa: E402
from src.utils import get_path, load_config, set_seed  # noqa: E402


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Carga el CSV crudo, renombra las columnas a los nombres reales (acepta alias "
            "dummiN), quita espacios a las categorías, verifica el catálogo contra el "
            "diccionario y guarda data/processed/base_estandar.csv."
        )
    )
    parser.add_argument("--sin-graficas", action="store_true", help="Este paso no genera gráficas; se acepta por uniformidad.")
    return parser.parse_args()


# ===== 2. Cálculo =====
def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla por columna: tipo, nulos, valores distintos, mínimo y máximo (numéricas)."""
    rows = []
    for col in df.columns:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        rows.append({
            "columna": col,
            "tipo": str(s.dtype),
            "nulos": int(s.isna().sum()),
            "valores_distintos": int(s.nunique()),
            "minimo": s.min() if numeric else "",
            "maximo": s.max() if numeric else "",
        })
    return pd.DataFrame(rows)


def catalog_vs_data(df: pd.DataFrame) -> pd.DataFrame:
    """Por dimensión (y familia): valores del catálogo presentes/ausentes y valores fuera de catálogo."""
    checks = [
        ("STRATEGY", "", None, cat.STRATEGIES),
        ("PRODUCT", "", None, cat.PRODUCTS),
        ("CHANNEL", "", None, cat.CHANNELS),
        ("AGE_TAG", "", None, cat.AGE_TAGS),
    ]
    for family in (cat.RECALIBRATION, cat.DARIA):
        checks.append(("STRATEGY_SEGMENT", family, family, cat.SEGMENTS_BY_FAMILY[family]))
        checks.append(("SCORE_TAG", family, family, cat.SCORES_BY_FAMILY[family]))

    family = df["STRATEGY"].astype(str).map(cat.STRATEGY_FAMILY)
    rows = []
    for col, label, fam, allowed in checks:
        values = df.loc[family == fam, col] if fam else df[col]
        present = set(values.astype(str).unique())
        rows.append({
            "dimension": col,
            "familia": label,
            "en_catalogo": len(allowed),
            "presentes_en_datos": len([v for v in allowed if v in present]),
            "ausentes_en_datos": "; ".join(v for v in allowed if v not in present),
            "fuera_de_catalogo": "; ".join(sorted(present - set(allowed))),
        })
    return pd.DataFrame(rows)


def compute(config: dict) -> dict:
    """Lee el diccionario, carga el CSV con nombres reales y arma las tablas de control."""
    # Qué hace: lee el diccionario y extrae sus catálogos tal cual (con espacios) para
    # compararlos con catalog.py. Por qué: catalog.py es la fuente que usa el resto del
    # código; si el diccionario cambia, este paso lo detecta en lugar de fallar más adelante.
    dictionary = read_dictionary(get_path(config, "dictionary"))
    dict_cat = dictionary_catalog(dictionary)
    alias_to_real = column_map(dictionary)

    # Qué hace: carga el CSV, renombra a nombres reales y aplica strip a las categorías.
    # Por qué: el diccionario y los datos traen valores con espacio al final; sin strip,
    # "Brands-NonPayroll-OffUs&LE " y "Brands-NonPayroll-OffUs&LE" serían categorías distintas.
    df, info = load_raw(get_path(config, "raw_data"), alias_to_real)

    return {
        "df": df,
        "info": info,
        "alias_to_real": alias_to_real,
        "catalog_diffs": catalog_differences(dict_cat),
        "dict_trailing": values_with_trailing_spaces(dict_cat),
        "profile": column_profile(df),
        "catalog_vs_data": catalog_vs_data(df),
    }


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda column_map.json, la base estándar y las tablas de control; devuelve las rutas."""
    written = []
    reports = get_path(config, "reports_dir") / "01_datos"
    reports.mkdir(parents=True, exist_ok=True)

    cmap_path = get_path(config, "contracts_dir") / "column_map.json"
    cmap = {
        "descripcion": "Mapa alias dummiN → nombre real (del diccionario). MOB no tiene alias.",
        "alias_a_nombre_real": results["alias_to_real"],
        "nombre_real_a_alias": {v: k for k, v in results["alias_to_real"].items()},
        "sin_alias": ["MOB"],
        "formato_encontrado_en_csv": results["info"]["formato"],
    }
    with open(cmap_path, "w", encoding="utf-8") as f:
        json.dump(cmap, f, ensure_ascii=False, indent=2)
        f.write("\n")
    written.append(cmap_path)

    base_path = get_path(config, "processed_dir") / "base_estandar.csv"
    results["df"].to_csv(base_path, index=False, encoding="utf-8")
    written.append(base_path)

    for name in ("profile", "catalog_vs_data"):
        out = reports / ("carga_perfil_columnas.csv" if name == "profile" else "carga_catalogo_vs_datos.csv")
        results[name].to_csv(out, index=False, encoding="utf-8")
        written.append(out)
    return written


# ===== 4. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 0 si todo salió bien y 1 si hubo una falla bloqueante."""
    parse_args()
    config = load_config()
    set_seed(config["seed"])

    try:
        results = compute(config)
    except (ColumnFormatError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    df, info = results["df"], results["info"]
    print("Carga del CSV:")
    print(f"  Formato de columnas encontrado: {info['formato']}")
    print(f"  Filas: {len(df):,}   Columnas: {df.shape[1]}   Memoria: {df.memory_usage(deep=True).sum() / 1e6:,.1f} MB")
    print(f"  Nulos totales: {int(df.isna().sum().sum()):,}")
    print(f"  Columnas faltantes del contrato: {info['faltantes'] or 'ninguna'}")
    print(f"  Columnas no declaradas (descartadas): {info['no_declaradas'] or 'ninguna'}")
    print("  Tipos:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:18s} {dtype}")
    print("Espacios en categorías:")
    print(f"  Valores del diccionario con espacio al final: {results['dict_trailing']}")
    changed = {k: v for k, v in info["filas_con_espacios"].items() if v}
    print(f"  Filas de datos normalizadas con strip: {changed or 'ninguna'}")
    print("Catálogo vs. diccionario:")
    if results["catalog_diffs"]:
        for d in results["catalog_diffs"]:
            print(f"  DIFERENCIA: {d}")
    else:
        print("  catalog.py coincide con el diccionario.")
    print("Catálogo vs. datos (valores del catálogo que no aparecen en los datos):")
    for _, row in results["catalog_vs_data"].iterrows():
        label = f"{row['dimension']}{'/' + row['familia'] if row['familia'] else ''}"
        print(f"  {label:30s} {row['presentes_en_datos']}/{row['en_catalogo']} presentes"
              f"{'; ausentes: ' + row['ausentes_en_datos'] if row['ausentes_en_datos'] else ''}"
              f"{'; FUERA DE CATÁLOGO: ' + row['fuera_de_catalogo'] if row['fuera_de_catalogo'] else ''}")

    written = save_results(results, config)
    print("Archivos escritos:")
    for path in written:
        print(f"  - {path}")

    if results["catalog_diffs"] or info["faltantes"]:
        print("FALLA: el catálogo no coincide con el diccionario o faltan columnas del contrato.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
