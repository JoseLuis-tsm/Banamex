"""Contrato del paquete de evidencia (ver docs/contratos/evidencia.md).

Define los campos obligatorios y una función que revisa si un paquete los cumple.
La usan las pruebas, el agente (para no redactar sobre un paquete mal formado) y,
más adelante, la sincronización entre ejemplos simulados y evidencia real.
"""

from typing import Any, Dict, List

# Catálogo de consultas: consulta_id -> nombre de la función de evidencia.
CONSULTAS = {
    "C1": "explain_profile",
    "C2": "variable_effect",
    "C3": "compare_groups",
    "C4": "suggest_strategy",
    "C5": "vintage_evolution",
    "C6": "summarize_batch",
    "C7": "define_term",
}

ESTADOS = ("ok", "no_evaluable", "sin_evidencia")

# Campo -> tipo esperado. "motivo" se revisa aparte (None o texto según el estado).
CAMPOS = {
    "consulta_id": str,
    "estado": str,
    "funcion": str,
    "parametros": dict,
    "cifras": dict,
    "tablas": list,
    "frases_base": list,
    "fragmentos": list,
    "versiones": dict,
    "simulado": bool,
}


def _is_number(value: Any) -> bool:
    """True si el valor es int o float (bool no cuenta como número)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_tables(tablas: List[Any]) -> List[str]:
    """Revisa que cada tabla tenga nombre, columnas y filas del mismo largo que las columnas."""
    errors = []
    for i, tabla in enumerate(tablas):
        if not isinstance(tabla, dict) or set(tabla) != {"nombre", "columnas", "filas"}:
            errors.append(f"tablas[{i}] debe tener exactamente las claves nombre, columnas y filas.")
            continue
        n_cols = len(tabla["columnas"])
        for j, fila in enumerate(tabla["filas"]):
            if not isinstance(fila, list) or len(fila) != n_cols:
                errors.append(f"tablas[{i}] ('{tabla['nombre']}'), fila {j}: debe tener {n_cols} valores.")
    return errors


def validate_evidence(paquete: Dict[str, Any]) -> List[str]:
    """Revisa un paquete de evidencia contra el contrato.

    Parámetros:
        paquete: diccionario devuelto por una función de evidencia (o leído de data/mock/).

    Devuelve:
        Lista de errores en español; vacía si el paquete cumple el contrato.
    """
    errors = []
    faltan = [c for c in list(CAMPOS) + ["motivo"] if c not in paquete]
    sobran = [c for c in paquete if c not in CAMPOS and c != "motivo"]
    if faltan:
        errors.append(f"Faltan campos: {', '.join(faltan)}.")
    if sobran:
        errors.append(f"Campos no declarados en el contrato: {', '.join(sobran)}.")
    for campo, tipo in CAMPOS.items():
        if campo in paquete and not isinstance(paquete[campo], tipo):
            errors.append(f"'{campo}' debe ser de tipo {tipo.__name__}.")
    if errors:
        return errors

    cid, estado = paquete["consulta_id"], paquete["estado"]
    if cid not in CONSULTAS:
        errors.append(f"consulta_id '{cid}' no está en el catálogo (C1 a C7).")
    elif paquete["funcion"] != f"evidence.{CONSULTAS[cid]}":
        errors.append(f"funcion debe ser 'evidence.{CONSULTAS[cid]}' para {cid}.")
    if estado not in ESTADOS:
        errors.append(f"estado '{estado}' no es válido; usa uno de {', '.join(ESTADOS)}.")

    motivo = paquete["motivo"]
    if estado == "ok" and motivo is not None:
        errors.append("Con estado 'ok', motivo debe ser null.")
    if estado != "ok" and not (isinstance(motivo, str) and motivo.strip()):
        errors.append(f"Con estado '{estado}', motivo debe explicar por qué (texto no vacío).")
    if estado != "ok" and (paquete["cifras"] or paquete["tablas"]):
        errors.append(f"Con estado '{estado}', cifras y tablas deben ir vacías (el agente se abstiene).")

    for nombre, valor in paquete["cifras"].items():
        if not (_is_number(valor) or isinstance(valor, (str, bool))):
            errors.append(f"cifras['{nombre}'] debe ser número, texto o booleano.")
    errors += _check_tables(paquete["tablas"])
    if not all(isinstance(f, str) for f in paquete["frases_base"]):
        errors.append("frases_base debe ser una lista de textos.")

    if cid != "C7" and paquete["fragmentos"]:
        errors.append("fragmentos solo se usa en C7.")
    for i, frag in enumerate(paquete["fragmentos"]):
        if not isinstance(frag, dict) or not {"fuente", "texto"} <= set(frag):
            errors.append(f"fragmentos[{i}] debe tener al menos fuente y texto.")
    if cid == "C7" and estado == "ok" and not paquete["fragmentos"]:
        errors.append("C7 con estado 'ok' debe traer al menos un fragmento.")
    return errors
