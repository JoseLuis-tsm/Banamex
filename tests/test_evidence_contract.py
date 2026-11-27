"""Pruebas de contrato de la evidencia: ejemplos simulados, get_evidence() y validador."""

import copy
import json
import re
from pathlib import Path

import pytest

from src.evidence import CONSULTAS, get_evidence, list_mock_names, validate_evidence
from src.evidence.service import _get_evidence, load_mocks
from src.utils import PROJECT_ROOT, get_path, load_config

MOCK_DIR = get_path(load_config(), "mock_dir")
MOCKS = load_mocks(MOCK_DIR)
DOC = PROJECT_ROOT / "docs" / "contratos" / "evidencia.md"


# ===== Ejemplos simulados =====
def test_hay_un_ejemplo_por_consulta_y_por_estado_especial():
    esperados = {f"evidencia_c{i}" for i in range(1, 8)} | {"evidencia_no_evaluable", "evidencia_sin_evidencia"}
    assert set(MOCKS) == esperados
    assert MOCKS["evidencia_no_evaluable"]["estado"] == "no_evaluable"
    assert MOCKS["evidencia_sin_evidencia"]["estado"] == "sin_evidencia"
    for i in range(1, 8):
        assert MOCKS[f"evidencia_c{i}"]["consulta_id"] == f"C{i}"
        assert MOCKS[f"evidencia_c{i}"]["estado"] == "ok"


@pytest.mark.parametrize("nombre", sorted(MOCKS))
def test_cada_ejemplo_cumple_el_contrato(nombre):
    assert validate_evidence(MOCKS[nombre]) == []


@pytest.mark.parametrize("nombre", sorted(MOCKS))
def test_cada_ejemplo_esta_marcado_como_simulado(nombre):
    assert MOCKS[nombre]["simulado"] is True


def _numbers_in_text(texto):
    """Números del texto que no forman parte de una palabra (excluye 'Over30', 'S3')."""
    return [float(m.replace(",", "")) for m in re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?!\w)", texto)]


def _backed_values(paquete):
    """Valores numéricos de cifras y tablas, también en porcentaje (x100, 1 decimal)."""
    valores = [v for v in paquete["cifras"].values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for tabla in paquete["tablas"]:
        for fila in tabla["filas"]:
            valores += [v for v in fila if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return valores + [round(v * 100, 1) for v in valores]


def _texts_in(value):
    """Todos los textos dentro de un valor anidado (nombres de catálogo como 'DARIA 1.6')."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _texts_in(v)]
    if isinstance(value, list):
        return [t for v in value for t in _texts_in(v)]
    return []


@pytest.mark.parametrize("nombre", sorted(MOCKS))
def test_cifras_de_frases_base_estan_respaldadas(nombre):
    paquete = MOCKS[nombre]
    respaldo = _backed_values(paquete)
    # Los números que forman parte de un nombre del paquete ("DARIA 1.6") no son cifras:
    # se quitan esos textos (los más largos primero) antes de buscar números.
    nombres = sorted(set(_texts_in([paquete["cifras"], paquete["tablas"], paquete["parametros"]])), key=len, reverse=True)
    for frase in paquete["frases_base"]:
        limpia = frase
        for texto in nombres:
            limpia = limpia.replace(texto, " ")
        for x in _numbers_in_text(limpia):
            assert any(abs(x - v) < 1e-9 for v in respaldo), f"{nombre}: {x} en '{frase}' no está en cifras ni tablas"


def test_ejemplos_del_documento_coinciden_con_data_mock():
    texto = DOC.read_text(encoding="utf-8")
    bloques = re.findall(r"<!-- ejemplo: (\w+) -->\s*```json\n(.*?)```", texto, flags=re.S)
    assert {n for n, _ in bloques} == set(MOCKS), "evidencia.md debe tener un ejemplo por archivo de data/mock/"
    for nombre, bloque in bloques:
        assert json.loads(bloque) == MOCKS[nombre], f"El ejemplo {nombre} del documento no coincide con data/mock/"


# ===== get_evidence() =====
@pytest.mark.parametrize("consulta_id", sorted(CONSULTAS))
def test_get_evidence_mock_devuelve_la_consulta(consulta_id):
    paquete = get_evidence(consulta_id)
    assert paquete["consulta_id"] == consulta_id
    assert paquete["estado"] == "ok"
    assert validate_evidence(paquete) == []


def test_get_evidence_con_parametros_exactos_devuelve_caso_especial():
    no_eval = MOCKS["evidencia_no_evaluable"]
    assert get_evidence(no_eval["consulta_id"], **no_eval["parametros"])["estado"] == "no_evaluable"
    sin_ev = MOCKS["evidencia_sin_evidencia"]
    assert get_evidence(sin_ev["consulta_id"], **sin_ev["parametros"])["estado"] == "sin_evidencia"


def test_get_evidence_devuelve_copia_independiente():
    paquete = get_evidence("C1")
    paquete["cifras"]["tasa_s3"] = -1
    assert get_evidence("C1")["cifras"]["tasa_s3"] != -1


def test_get_evidence_modo_real_no_implementado():
    config = copy.deepcopy(load_config())
    config["evidence_mode"] = "real"
    with pytest.raises(NotImplementedError, match="Fase 5A"):
        _get_evidence("C1", {}, config)


def test_get_evidence_errores_claros():
    with pytest.raises(ValueError):
        get_evidence("C9")
    config = copy.deepcopy(load_config())
    config["evidence_mode"] = "otro"
    with pytest.raises(ValueError):
        _get_evidence("C1", {}, config)


def test_list_mock_names():
    assert len(list_mock_names()) == 9


# ===== Validador con paquetes rotos a propósito =====
def _roto(**cambios):
    paquete = copy.deepcopy(MOCKS["evidencia_c3"])
    for k, v in cambios.items():
        if v is None and k != "motivo":
            paquete.pop(k)
        else:
            paquete[k] = v
    return paquete


@pytest.mark.parametrize(
    "paquete",
    [
        _roto(cifras=None),                           # falta un campo
        _roto(extra=1),                               # campo no declarado
        _roto(estado="aprobado"),                     # estado fuera de catálogo
        _roto(consulta_id="C9"),                      # consulta fuera de catálogo
        _roto(funcion="evidence.otra"),               # función que no corresponde
        _roto(motivo="no debería haber motivo"),      # ok con motivo
        _roto(estado="sin_evidencia"),                # sin motivo y con cifras
        _roto(tablas=[{"nombre": "t", "columnas": ["a", "b"], "filas": [[1]]}]),  # fila corta
        _roto(simulado="sí"),                         # tipo incorrecto
        _roto(fragmentos=[{"fuente": "x", "texto": "y"}]),  # fragmentos fuera de C7
    ],
)
def test_validador_detecta_paquetes_rotos(paquete):
    assert validate_evidence(paquete) != []
