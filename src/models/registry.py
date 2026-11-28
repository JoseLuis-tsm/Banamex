"""Registro de modelos: carpetas versionadas, metadata.json y actual.json (CLAUDE.md §7).

Estructura: models/<salida>/<versión>/{modelo.*, metadata.json} y models/<salida>/actual.json.
La versión es "<salida>-<hash>", con el hash de los datos, las variables y los
hiperparámetros: correr de nuevo con lo mismo reescribe la misma versión en lugar de
crear una nueva.
"""

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

LIBRARIES = ["numpy", "pandas", "scikit-learn", "xgboost", "optuna", "shap", "plotly"]


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Hash SHA-256 de un archivo (versión de los datos)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def library_versions() -> Dict[str, str]:
    """Versiones de Python y de las librerías del plan instaladas."""
    out = {"python": platform.python_version()}
    for lib in LIBRARIES:
        try:
            out[lib] = importlib_metadata.version(lib)
        except importlib_metadata.PackageNotFoundError:
            out[lib] = "no instalada"
    return out


def version_id(output: str, payload: Dict[str, Any]) -> str:
    """Versión determinista: '<salida>-' + 10 caracteres del SHA-256 del payload en JSON."""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return f"{output}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:10]}"


def write_json(path: Path, data: Dict[str, Any]) -> Path:
    """Escribe un JSON en UTF-8, con sangría y sin escapar acentos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
    return path


def read_json(path: Path) -> Dict[str, Any]:
    """Lee un JSON en UTF-8."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def version_dir(models_dir: Path, output: str, version: str) -> Path:
    """Carpeta de una versión: models/<salida>/<versión>/ (la crea si no existe)."""
    d = Path(models_dir) / output / version
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_metadata(models_dir: Path, output: str, version: str, metadata: Dict[str, Any]) -> Path:
    """Guarda metadata.json de la versión, agregando fecha (UTC) y versiones de librerías."""
    full = {
        "salida": output,
        "version": version,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versiones_librerias": library_versions(),
        **metadata,
    }
    return write_json(version_dir(models_dir, output, version) / "metadata.json", full)


def publish(models_dir: Path, output: str, version: str) -> Path:
    """Marca una versión como la publicada: escribe models/<salida>/actual.json."""
    return write_json(Path(models_dir) / output / "actual.json", {"salida": output, "version": version, "ruta": f"{output}/{version}"})


def current(models_dir: Path, output: str) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Carpeta y metadata de la versión publicada; None si no hay actual.json."""
    pointer = Path(models_dir) / output / "actual.json"
    if not pointer.exists():
        return None
    d = Path(models_dir) / read_json(pointer)["ruta"]
    return d, read_json(d / "metadata.json")


def vintage_list(frame: pd.DataFrame, split: Optional[str] = None) -> List[int]:
    """Vintages (enteros, ordenados) de un conjunto de la partición, o de todo el frame si split es None."""
    part = frame if split is None else frame[frame["CONJUNTO"] == split]
    return sorted(int(v) for v in part["VINTAGE"].unique())
