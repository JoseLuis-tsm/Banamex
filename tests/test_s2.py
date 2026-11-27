"""Pruebas de S2 (src/models/s2.py): cuentas planeadas × S1, exacto."""

import numpy as np
import pytest

from src.models.s2 import expected_total_amount


def test_s2_es_exactamente_cuentas_por_linea():
    assert expected_total_amount(1000, 38318.5) == 1000 * 38318.5
    assert expected_total_amount(0, 50000.0) == 0.0
    cuentas, lineas = np.array([3, 10, 250]), np.array([12345.67, 8800.0, 480000.0])
    assert np.array_equal(expected_total_amount(cuentas, lineas), cuentas * lineas)


@pytest.mark.parametrize("cuentas, linea", [(-1, 1000.0), (2.5, 1000.0), (10, 0.0), (10, -5.0), (10, float("nan"))])
def test_s2_rechaza_entradas_invalidas(cuentas, linea):
    with pytest.raises(ValueError):
        expected_total_amount(cuentas, linea)
