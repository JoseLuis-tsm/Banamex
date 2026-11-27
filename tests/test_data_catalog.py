"""Pruebas del catálogo del contrato de datos (src/data/catalog.py y dictionary.py)."""

import pytest

from src.data import catalog as cat
from src.data.dictionary import (
    catalog_differences, column_map, dictionary_catalog, read_dictionary, values_with_trailing_spaces,
)
from src.utils import get_path, load_config

DICTIONARY = read_dictionary(get_path(load_config(), "dictionary"))
DICT_CATALOG = dictionary_catalog(DICTIONARY)


def test_catalog_coincide_con_el_diccionario():
    assert catalog_differences(DICT_CATALOG) == []


def test_catalog_detecta_un_diccionario_distinto():
    alterado = dict(DICT_CATALOG, products=DICT_CATALOG["products"][:-1])
    assert any(d.startswith("PRODUCT") for d in catalog_differences(alterado))


def test_valores_con_espacio_al_final_del_diccionario():
    assert values_with_trailing_spaces(DICT_CATALOG) == ["Brands-NonPayroll-OffUs&LE ", "AAC Off Us Non-Payroll "]
    # catalog.py los guarda ya sin espacios
    todos = cat.PRODUCTS + cat.CHANNELS + cat.AGE_TAGS + sum(cat.SEGMENTS_BY_FAMILY.values(), [])
    assert all(v == v.strip() for v in todos)


def test_column_map_tiene_21_alias_y_mob_sin_alias():
    cmap = column_map(DICTIONARY)
    assert len(cmap) == 21
    assert "MOB" not in cmap.values()
    assert set(cmap.values()) | {"MOB"} == set(cat.ALL_COLUMNS)


def test_tamano_de_familias():
    assert len(cat.SEGMENTS_BY_FAMILY[cat.RECALIBRATION]) == 7
    assert len(cat.SEGMENTS_BY_FAMILY[cat.DARIA]) == 17
    assert len(cat.SCORES_BY_FAMILY[cat.RECALIBRATION]) == 8
    assert len(cat.SCORES_BY_FAMILY[cat.DARIA]) == 9


@pytest.mark.parametrize("vintage, esperado", [(202401, 32), (202402, 31), (202603, 6), (202608, 1), (202609, 0), (202512, 9)])
def test_max_mob(vintage, esperado):
    assert cat.max_mob(vintage) == esperado


@pytest.mark.parametrize(
    "vintage, estrategias, tipo",
    [
        (202401, ("14.Recalibration 1.0",), "determinista"),
        (202405, ("14.Recalibration 1.0",), "determinista"),
        (202406, ("15.Recalibration 2.0",), "determinista"),
        (202507, ("15.Recalibration 2.0",), "determinista"),
        (202508, ("r.Reca champ",), "determinista"),
        (202511, ("r.Reca champ", "DARIA 1.6"), "aleatoria"),
        (202607, ("r.Reca champ", "DARIA 1.6"), "aleatoria"),
        (202609, ("r.Reca champ", "DARIA 2.0"), "aleatoria"),
    ],
)
def test_estrategia_por_vintage(vintage, estrategias, tipo):
    assert cat.allowed_strategies(vintage) == estrategias
    assert cat.assignment_type(vintage) == tipo


def test_vintage_fuera_de_ventanas():
    assert cat.allowed_strategies(202312) is None
    assert cat.assignment_type(202610) is None


def test_score_position_va_de_0_a_1_en_orden():
    for family, scores in cat.SCORES_BY_FAMILY.items():
        posiciones = [cat.score_position(s, family) for s in scores]
        assert posiciones[0] == 0.0 and posiciones[-1] == 1.0
        assert posiciones == sorted(posiciones)
    assert cat.score_position("151-160", cat.RECALIBRATION) == pytest.approx(3 / 7)
    with pytest.raises(ValueError):
        cat.score_position("151-160", cat.DARIA)


def test_age_ordinal_y_mes_de_booking():
    assert [cat.age_ordinal(a) for a in cat.AGE_TAGS] == [1, 2, 3]
    with pytest.raises(ValueError):
        cat.age_ordinal("04| Otro")
    assert cat.booking_month(202512) == 12 and cat.booking_month(202601) == 1


def test_familia_de_estrategia():
    assert cat.family_of("r.Reca champ") == cat.RECALIBRATION
    assert cat.family_of("DARIA 2.0") == cat.DARIA
    assert cat.family_of("otra") is None
