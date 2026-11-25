"""Punto único de acceso a la evidencia: get_evidence().

Con evidence_mode = "mock" (configs/config.json) devuelve los ejemplos simulados de
data/mock/; con "real" todavía no hay funciones (llegan en la Fase 5A).
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evidence.contract import CONSULTAS
from src.utils import get_path, load_config


def load_mocks(mock_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Lee todos los ejemplos evidencia_*.json de la carpeta.

    Devuelve:
        Diccionario nombre de archivo (sin .json) -> paquete.
    """
    mocks = {}
    for path in sorted(Path(mock_dir).glob("evidencia_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            mocks[path.stem] = json.load(f)
    return mocks


def pick_mock(consulta_id: str, params: Dict[str, Any], mocks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Elige el ejemplo simulado para una consulta.

    Regla: si algún ejemplo de esa consulta tiene exactamente los mismos parámetros,
    se devuelve ese (así se piden los casos no_evaluable y sin_evidencia); si no,
    se devuelve el ejemplo base evidencia_cN.json, con sus propios parámetros, porque
    sus cifras corresponden a esos parámetros y no a los pedidos.
    """
    same_query = [m for m in mocks.values() if m["consulta_id"] == consulta_id]
    for mock in same_query:
        if params and mock["parametros"] == params:
            return copy.deepcopy(mock)
    default_name = f"evidencia_{consulta_id.lower()}"
    if default_name not in mocks:
        raise FileNotFoundError(f"No existe el ejemplo simulado {default_name}.json en data/mock/.")
    return copy.deepcopy(mocks[default_name])


def get_evidence(consulta_id: str, **params: Any) -> Dict[str, Any]:
    """Devuelve el paquete de evidencia de una consulta del catálogo (C1 a C7).

    Parámetros:
        consulta_id: identificador de la consulta, por ejemplo "C3".
        **params: parámetros de la consulta (por ejemplo dimension, valor_a, valor_b).

    Devuelve:
        Paquete de evidencia con el formato de docs/contratos/evidencia.md.

    Lanza:
        ValueError si consulta_id no está en el catálogo o evidence_mode no es válido.
        NotImplementedError con evidence_mode = "real" (la evidencia real llega en la Fase 5A).
    """
    return _get_evidence(consulta_id, params, load_config())


def _get_evidence(consulta_id: str, params: Dict[str, Any], config: Dict[str, Any],
                  mock_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Igual que get_evidence(), con la configuración explícita (útil en pruebas)."""
    if consulta_id not in CONSULTAS:
        raise ValueError(f"consulta_id '{consulta_id}' no está en el catálogo; usa uno de {', '.join(CONSULTAS)}.")
    mode = config.get("evidence_mode")
    if mode == "mock":
        mocks = load_mocks(mock_dir if mock_dir is not None else get_path(config, "mock_dir"))
        return pick_mock(consulta_id, params, mocks)
    if mode == "real":
        raise NotImplementedError(
            f"La evidencia real de {consulta_id} ({CONSULTAS[consulta_id]}) todavía no existe: "
            "se implementa en la Fase 5A. Mientras tanto usa evidence_mode = \"mock\" en configs/config.json."
        )
    raise ValueError(f"evidence_mode '{mode}' no es válido; usa \"mock\" o \"real\" en configs/config.json.")


def list_mock_names(mock_dir: Optional[Path] = None) -> List[str]:
    """Lista los nombres de los ejemplos simulados disponibles."""
    return list(load_mocks(mock_dir if mock_dir is not None else get_path(load_config(), "mock_dir")))
