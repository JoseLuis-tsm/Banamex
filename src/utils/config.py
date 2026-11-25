"""Lectura de la configuración del proyecto (configs/config.json)."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Raíz del repositorio: src/utils/config.py -> sube dos niveles hasta la carpeta del repo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.json"


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Lee el archivo de configuración JSON en UTF-8.

    Parámetros:
        path: ruta del JSON. Si es None, usa configs/config.json del repo.

    Devuelve:
        Diccionario con la configuración tal como está en el archivo.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_path(config: Dict[str, Any], key: str) -> Path:
    """Devuelve como ruta absoluta una entrada de config["paths"].

    Las rutas del JSON son relativas a la raíz del repo, así funcionan igual
    desde un notebook, desde pytest o desde la API, sin importar la carpeta actual.

    Parámetros:
        config: configuración cargada con load_config().
        key: nombre de la ruta (por ejemplo "raw_data" o "models_dir").

    Devuelve:
        Ruta absoluta (Path). Lanza KeyError si la clave no existe.
    """
    return PROJECT_ROOT / config["paths"][key]
