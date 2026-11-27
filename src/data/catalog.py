"""Catálogo del contrato de datos: columnas, valores permitidos y reglas por vintage.

Fuente: data/contracts/diccionario_variables.csv. Los valores se guardan **sin espacios
al final** (el diccionario y el CSV traen, por ejemplo, "Brands-NonPayroll-OffUs&LE ";
el cargador aplica .str.strip() a las dimensiones de texto). La prueba
tests/test_data_catalog.py comprueba que este archivo coincide con el diccionario.
"""

from typing import Dict, List, Optional, Tuple

# ===== Columnas =====
# Llave única de una fila: las 7 dimensiones de la cohorte más el mes de desempeño.
COHORT_COLUMNS: List[str] = [
    "VINTAGE", "STRATEGY", "STRATEGY_SEGMENT", "PRODUCT", "CHANNEL", "AGE_TAG", "SCORE_TAG",
]
KEY_COLUMNS: List[str] = COHORT_COLUMNS + ["MOB"]

# Dimensiones de texto (se normalizan con strip y se guardan como categóricas).
TEXT_DIMENSIONS: List[str] = ["STRATEGY", "STRATEGY_SEGMENT", "PRODUCT", "CHANNEL", "AGE_TAG", "SCORE_TAG"]

# Volumen y exposición: sirven de pesos y denominadores, nunca de variables de entrada.
VOLUME_COLUMNS: List[str] = ["ACCOUNTS", "CREDITLINESAMTS", "ENRACCTS", "ENRAMTS"]

# Resultado: solo construyen objetivos, nunca son variables de entrada (CLAUDE.md §5).
RESULT_COLUMNS: List[str] = [
    "OVER30ACCTS", "OVER30AMTS", "GCLAMTS", "NCLAMTS", "RECOVERYAMTS", "REVENUE",
    "EXPENSES_ACQ", "EXPENSES_ECM_VAR", "EXPENSES_ECM_FIX", "EXPENSES_TOTAL",
]

NUMERIC_COLUMNS: List[str] = ["VINTAGE", "MOB"] + VOLUME_COLUMNS + RESULT_COLUMNS
ALL_COLUMNS: List[str] = KEY_COLUMNS + VOLUME_COLUMNS + RESULT_COLUMNS

# Columnas que el diccionario declara como enteras (conteos y línea de crédito).
INTEGER_COLUMNS: List[str] = ["VINTAGE", "MOB", "ACCOUNTS", "CREDITLINESAMTS", "ENRACCTS", "OVER30ACCTS"]

# Columnas que se esperan acumuladas dentro de una cohorte (confirmadas por Luis).
EXPECTED_CUMULATIVE: List[str] = ["OVER30ACCTS", "REVENUE"]

# ===== Estrategias y familias =====
RECALIBRATION = "Recalibration"
DARIA = "DARIA"

STRATEGY_FAMILY: Dict[str, str] = {
    "14.Recalibration 1.0": RECALIBRATION,
    "15.Recalibration 2.0": RECALIBRATION,
    "r.Reca champ": RECALIBRATION,
    "DARIA 1.6": DARIA,
    "DARIA 2.0": DARIA,
}
STRATEGIES: List[str] = list(STRATEGY_FAMILY)

# DARIA 2.0 solo tiene MOB 0 y 1: tiene línea asignada, pero no desempeño de riesgo.
NOT_RISK_EVALUABLE: List[str] = ["DARIA 2.0"]

SEGMENTS_BY_FAMILY: Dict[str, List[str]] = {
    RECALIBRATION: [
        "Brands-NonPayroll-OffUs&LE", "Brands-NonPayroll-OnUs", "Brands-Payroll-OffUs&LE",
        "Brands-Payroll-OnUs", "Costco", "HomeDepot & Affinity", "Priority & Affluent",
    ],
    DARIA: [
        "AAC Off Us Non-Payroll", "AAC Off Us Payroll", "AAC On Us", "Affinity", "Affluent",
        "Costco - Young <31", "Costco - Non-LE", "Costco - LE", "Home Depot - Young <31",
        "Home Depot - LE", "Home Depot - Non-LE", "LE Commun. & NTR", "LE Inq Only & No Hit",
        "LE Thin", "Priority Qualified", "Young < 25", "Young 25-30",
    ],
}

# Rangos de score ordenados de menor a mayor score (el orden importa: de él sale la
# posición relativa 0-1 que usa la Fase 2).
SCORES_BY_FAMILY: Dict[str, List[str]] = {
    RECALIBRATION: ["<130", "130-140", "141-150", "151-160", "161-170", "171-180", "181-190", "190+"],
    DARIA: [
        "a.<=154", "b.155-159", "c.160-164", "d.165-169", "e.170-174",
        "f.175-189", "g.190-199", "h.200-204", "i.205+",
    ],
}

# ===== Dimensiones compartidas =====
PRODUCTS: List[str] = [
    "Affinity", "Clasica", "Conquista", "Costco", "Descubre", "Explora",
    "Home Depot", "Joy", "Oro", "Platinum", "Teleton", "Beyond",
]
CHANNELS: List[str] = [
    "Aeropuerto", "Affinity", "Beyond/Prestige", "Branch", "Costco", "F2F",
    "Home Depot", "Internet", "One Click", "Others", "TMK", "Turismo",
]
# Ordenados de menor a mayor edad (de aquí sale el ordinal 1 a 3).
AGE_TAGS: List[str] = ["01| Young < 25", "02| Young 25-30", "03| Mature >30"]

# ===== Estrategia según vintage =====
# (vintage inicial, vintage final, estrategias permitidas, tipo de asignación)
STRATEGY_WINDOWS: List[Tuple[int, int, Tuple[str, ...], str]] = [
    (202401, 202405, ("14.Recalibration 1.0",), "determinista"),
    (202406, 202507, ("15.Recalibration 2.0",), "determinista"),
    (202508, 202510, ("r.Reca champ",), "determinista"),
    (202511, 202607, ("r.Reca champ", "DARIA 1.6"), "aleatoria"),
    (202608, 202609, ("r.Reca champ", "DARIA 2.0"), "aleatoria"),
]

# ===== Rangos del diccionario =====
VINTAGE_MIN, VINTAGE_MAX = 202401, 202609
MOB_MIN, MOB_MAX = 0, 32
AMOUNT_MIN, AMOUNT_MAX = 0, 99999999999999  # mismo rango para volumen y resultado


def vintage_to_index(vintage: int) -> int:
    """Convierte AAAAMM a un número de meses (año*12 + mes) para restar fechas."""
    return (int(vintage) // 100) * 12 + int(vintage) % 100


def max_mob(vintage: int, cutoff: int = VINTAGE_MAX) -> int:
    """MOB máximo observable de una vintage con el corte estático (202609 → 0, 202401 → 32)."""
    return vintage_to_index(cutoff) - vintage_to_index(vintage)


def booking_month(vintage: int) -> int:
    """Mes del booking (1 a 12) derivado de la vintage AAAAMM."""
    return int(vintage) % 100


def allowed_strategies(vintage: int) -> Optional[Tuple[str, ...]]:
    """Estrategias permitidas para una vintage; None si la vintage está fuera de las ventanas."""
    for start, end, strategies, _ in STRATEGY_WINDOWS:
        if start <= int(vintage) <= end:
            return strategies
    return None


def assignment_type(vintage: int) -> Optional[str]:
    """'determinista' o 'aleatoria' según la ventana de la vintage; None si está fuera."""
    for start, end, _, kind in STRATEGY_WINDOWS:
        if start <= int(vintage) <= end:
            return kind
    return None


def family_of(strategy: str) -> Optional[str]:
    """Familia (Recalibration o DARIA) de una estrategia; None si no está en el catálogo."""
    return STRATEGY_FAMILY.get(strategy)


def score_position(score_tag: str, family: str) -> float:
    """Posición relativa del rango de score dentro de su familia, entre 0 y 1.

    Fórmula: posición = i / (n − 1), donde i es el índice del rango en
    SCORES_BY_FAMILY[family] (0 = score más bajo) y n el número de rangos de la familia.
    Lanza ValueError si el rango no pertenece a la familia.
    """
    scores = SCORES_BY_FAMILY[family]
    if score_tag not in scores:
        raise ValueError(f"El rango de score '{score_tag}' no pertenece a la familia {family}.")
    return scores.index(score_tag) / (len(scores) - 1)


def age_ordinal(age_tag: str) -> int:
    """Edad como ordinal 1 a 3 (1 = menor de 25). Lanza ValueError si no está en el catálogo."""
    if age_tag not in AGE_TAGS:
        raise ValueError(f"El rango de edad '{age_tag}' no está en el catálogo.")
    return AGE_TAGS.index(age_tag) + 1
