"""S2: monto total esperado. Es un cálculo determinista, no un modelo (CLAUDE.md §6)."""

from typing import Union

import numpy as np

Number = Union[int, float, np.ndarray]


def expected_total_amount(planned_accounts: Number, expected_line: Number) -> Number:
    """Monto total esperado de colocar N cuentas.

    Fórmula: S2 = cuentas planeadas × S1, con S1 la línea promedio esperada por cuenta.

    Parámetros:
        planned_accounts: cuentas planeadas (entero o arreglo, ≥ 0).
        expected_line: línea promedio esperada por cuenta (S1, > 0).

    Devuelve:
        El producto exacto, con la misma forma que las entradas. Lanza ValueError si hay
        cuentas negativas, no enteras o una línea ≤ 0.
    """
    accounts = np.asarray(planned_accounts, dtype=float)
    line = np.asarray(expected_line, dtype=float)
    if np.any(accounts < 0) or np.any(accounts != np.floor(accounts)):
        raise ValueError("Las cuentas planeadas deben ser enteros mayores o iguales a 0.")
    if np.any(line <= 0) or np.any(~np.isfinite(line)):
        raise ValueError("La línea esperada (S1) debe ser un número positivo.")
    result = accounts * line
    return float(result) if result.ndim == 0 else result
