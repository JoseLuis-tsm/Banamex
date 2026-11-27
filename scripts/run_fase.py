"""Corre en orden los scripts de una fase.

Objetivo:
    Ejecutar todos los scripts numerados de una fase (scripts/<NN>_*/NN_*.py),
    uno tras otro, deteniéndose en el primer error.

Entradas:
    - La fase a correr: 1, 2, 3, 4, 5a, 5b o 6 (también acepta 01, 05a, etc.).

Salidas:
    - Lo que escriba cada script, más un resumen en la terminal
      (script, resultado y segundos).

Cómo correrlo (desde la raíz del repo):
    python scripts/run_fase.py 1
    python scripts/run_fase.py 1 --desde 03
    python scripts/run_fase.py 2 --sin-graficas

Listo cuando:
    Todos los scripts de la fase terminan con código 0. Si la fase todavía no
    tiene scripts, avisa y termina sin error.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

# Un script de fase se llama NN_nombre.py (dos dígitos, guion bajo, nombre).
SCRIPT_PATTERN = re.compile(r"^(\d{2})_.+\.py$")


def normalize_phase(phase: str) -> str:
    """Convierte '1', '01', '5A' o '05a' al prefijo de carpeta ('01', '05a')."""
    match = re.fullmatch(r"0*(\d)([abAB]?)", phase.strip())
    if not match:
        raise ValueError(f"Fase no válida: '{phase}'. Usa 1, 2, 3, 4, 5a, 5b o 6.")
    return f"0{match.group(1)}{match.group(2).lower()}"


def find_phase_dir(phase: str, scripts_dir: Path = SCRIPTS_DIR) -> Optional[Path]:
    """Devuelve la carpeta scripts/<prefijo>_* de la fase, o None si no existe."""
    prefix = normalize_phase(phase) + "_"
    matches = sorted(d for d in scripts_dir.iterdir() if d.is_dir() and d.name.startswith(prefix))
    return matches[0] if matches else None


def list_phase_scripts(phase_dir: Path, start: Optional[str] = None) -> List[Path]:
    """Lista en orden los scripts NN_*.py de la carpeta, desde el número `start` si se da.

    Los archivos que empiezan con "_" (como la plantilla) no cumplen el patrón y se ignoran.
    """
    scripts = sorted(p for p in phase_dir.iterdir() if p.is_file() and SCRIPT_PATTERN.match(p.name))
    if start is not None:
        start_num = int(start)
        scripts = [p for p in scripts if int(SCRIPT_PATTERN.match(p.name).group(1)) >= start_num]
    return scripts


def run_script(script: Path, extra_args: List[str]) -> int:
    """Corre un script con el mismo Python, desde la raíz del repo, y devuelve su código de salida."""
    return subprocess.run([sys.executable, str(script)] + extra_args, cwd=str(ROOT)).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corre en orden los scripts de una fase y se detiene en el primer error."
    )
    parser.add_argument("fase", help="Fase a correr: 1, 2, 3, 4, 5a, 5b o 6.")
    parser.add_argument("--desde", metavar="NN", help="Empieza en el script con ese número (ej. 03).")
    parser.add_argument(
        "--sin-graficas", action="store_true", help="Pasa --sin-graficas a cada script."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        phase_dir = find_phase_dir(args.fase)
    except ValueError as err:
        print(f"Error: {err}")
        return 2
    if phase_dir is None:
        print(f"Error: no existe la carpeta de la fase {args.fase} en {SCRIPTS_DIR}.")
        return 2
    if args.desde is not None and not args.desde.isdigit():
        print(f"Error: --desde debe ser un número de script, por ejemplo 03 (recibí '{args.desde}').")
        return 2

    scripts = list_phase_scripts(phase_dir, args.desde)
    rel_dir = phase_dir.relative_to(ROOT).as_posix()
    if not scripts:
        desde = f" a partir del {args.desde}" if args.desde else ""
        print(f"Aviso: la fase {args.fase} ({rel_dir}) no tiene scripts{desde} todavía. Nada que correr.")
        return 0

    extra = ["--sin-graficas"] if args.sin_graficas else []
    print(f"Fase {args.fase}: {len(scripts)} script(s) en {rel_dir}")
    summary = []
    failed = False
    for script in scripts:
        print(f"\n===== {script.name} =====", flush=True)
        t0 = time.perf_counter()
        code = run_script(script, extra)
        summary.append((script.name, code, time.perf_counter() - t0))
        if code != 0:
            failed = True
            break

    print("\nResumen:")
    for name, code, secs in summary:
        status = "OK" if code == 0 else f"FALLÓ (código {code})"
        print(f"  {name:<45} {status:<20} {secs:8.1f} s")
    pending = [s.name for s in scripts[len(summary):]]
    if failed:
        print(f"  Se detuvo en el primer error; no se corrieron: {', '.join(pending) or 'ninguno'}.")
        num = SCRIPT_PATTERN.match(summary[-1][0]).group(1)
        print(f"  Para continuar después de corregir: python scripts/run_fase.py {args.fase} --desde {num}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
