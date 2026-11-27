"""NN — Título corto del paso.

Objetivo:
    Qué resuelve este script, en una o dos frases.

Entradas:
    - Archivos que lee (rutas tomadas de configs/config.json).
    - Funciones de src/ que usa.

Salidas:
    - Archivos que escribe (data/processed/, reports/<fase>/ o models/).

Cómo correrlo (desde la raíz del repo):
    python scripts/<fase>/NN_nombre.py
    python scripts/<fase>/NN_nombre.py --sin-graficas

Listo cuando:
    La condición verificable que indica que el paso terminó bien.

Cómo usar esta plantilla:
    Copia este archivo a scripts/<fase>/NN_nombre.py, cambia el docstring y
    rellena las secciones. run_fase.py ignora los archivos que empiezan con "_".
"""

# Encabezado estándar: agrega la raíz del repo a sys.path para poder importar src
# sin importar desde qué carpeta se corra el script. parents[2] supone que el archivo
# está en scripts/<fase>/; por eso la plantilla funciona una vez copiada ahí, no en scripts/.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402

from src.utils import get_path, load_config, set_seed  # noqa: E402


# ===== 1. Argumentos =====
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="NN — Título corto del paso. (Explica aquí qué hace el script.)"
    )
    parser.add_argument(
        "--sin-graficas",
        action="store_true",
        help="No genera las gráficas HTML (solo cálculos y tablas).",
    )
    return parser.parse_args()


# ===== 2. Cálculo =====
def compute(config: dict) -> dict:
    """Hace el cálculo principal y devuelve los resultados.

    Parámetros:
        config: configuración cargada con load_config().

    Devuelve:
        Diccionario con los resultados que se guardan y se grafican.
    """
    # Qué hace: ... (variables, funciones, cómo).
    # Por qué: ... (el fundamento).
    return {}


# ===== 3. Guardado =====
def save_results(results: dict, config: dict) -> list:
    """Guarda las tablas en CSV (UTF-8) y devuelve la lista de archivos escritos."""
    written = []
    # Ejemplo: out = get_path(config, "reports_dir") / "<fase>" / "tabla.csv"
    #          out.parent.mkdir(parents=True, exist_ok=True)
    #          tabla.to_csv(out, index=False, encoding="utf-8"); written.append(out)
    return written


# ===== 4. Gráficas (funciones aparte, se omiten con --sin-graficas) =====
def make_charts(results: dict, config: dict) -> list:
    """Construye las gráficas con Plotly y las guarda en HTML con fig.write_html()."""
    written = []
    return written


# ===== 5. Programa principal =====
def main() -> int:
    """Corre el paso completo. Devuelve 0 si todo salió bien y 1 si hubo una falla bloqueante."""
    args = parse_args()
    config = load_config()
    set_seed(config["seed"])

    results = compute(config)
    written = save_results(results, config)
    if not args.sin_graficas:
        written += make_charts(results, config)

    # Resumen claro en la terminal: qué hizo y qué archivos escribió.
    print("Resumen:")
    print(f"  Archivos escritos: {len(written)}")
    for path in written:
        print(f"    - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
