"""Pruebas de S4 (src/strategy/s4.py): ventana, soporte común y bootstrap."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.s4 import (
    CHALLENGER, CHAMPION, STATUS_COMMON, STATUS_NO_EVIDENCE, bootstrap_differences, build_window, profile_table,
)


def _cohorte(vintage, strategy, product, enr, o30, seg="x", score="s"):
    return {"VINTAGE": vintage, "STRATEGY": strategy, "STRATEGY_SEGMENT": seg, "PRODUCT": product, "CHANNEL": "Branch",
            "AGE_TAG": "03| Mature >30", "SCORE_TAG": score, "ACCOUNTS": enr + 2, "ENRACCTS": enr, "OVER30ACCTS": o30}


@pytest.fixture
def s3():
    rows = []
    for v in (202511, 202512, 202601):
        rows += [_cohorte(v, CHAMPION, "Oro", 20, 1), _cohorte(v, CHALLENGER, "Oro", 20, 4)]   # DARIA peor
        rows += [_cohorte(v, CHAMPION, "Costco", 20, 2)]                                      # solo champion
    rows += [_cohorte(202510, CHAMPION, "Oro", 20, 10), _cohorte(202601, "DARIA 2.0", "Oro", 20, 0)]  # fuera de ventana / no evaluable
    rows += [_cohorte(202512, CHAMPION, "Joy", 10, 1), _cohorte(202512, CHALLENGER, "Joy", 10, 1)]   # una cohorte por lado
    return pd.DataFrame(rows)


def test_ventana_excluye_fuera_de_rango_y_daria_2(s3):
    w = build_window(s3)
    assert set(w["STRATEGY"]) == {CHAMPION, CHALLENGER}
    assert w["VINTAGE"].min() >= 202511
    assert not w.loc[w["PRODUCT"] == "Costco", "SOPORTE_COMUN"].any()
    assert w.loc[w["PRODUCT"] == "Oro", "SOPORTE_COMUN"].all()


def test_tabla_por_perfil_y_estado(s3):
    p = profile_table(build_window(s3)).set_index("PRODUCT")
    assert p.loc["Costco", "estado"] == STATUS_NO_EVIDENCE and p.loc["Costco", "cohortes_daria"] == 0
    assert p.loc["Oro", "estado"] == STATUS_COMMON
    assert p.loc["Oro", "tasa_daria"] == pytest.approx(12 / 60) and p.loc["Oro", "tasa_champ"] == pytest.approx(3 / 60)


def test_bootstrap_reproducible_y_concluyente(s3):
    w = build_window(s3)
    a, b = bootstrap_differences(w, seed=42, n_boot=500), bootstrap_differences(w, seed=42, n_boot=500)
    assert a.equals(b)
    oro = a.set_index("PERFIL").loc["Oro | Branch | 03| Mature >30"]
    assert oro["diferencia"] == pytest.approx(12 / 60 - 3 / 60)
    # Todas las cohortes de cada lado tienen la misma tasa: el intervalo colapsa en la diferencia
    assert oro["ic_inferior"] == pytest.approx(oro["diferencia"]) and oro["concluyente"]


def test_una_sola_cohorte_no_es_estimable_ni_concluyente(s3):
    joy = bootstrap_differences(build_window(s3), seed=42, n_boot=200).set_index("PERFIL").loc["Joy | Branch | 03| Mature >30"]
    assert not joy["estimable"] and not joy["concluyente"] and np.isnan(joy["ic_inferior"])


def test_intervalo_contiene_la_diferencia_con_ruido():
    rng = np.random.default_rng(0)
    rows = [_cohorte(202512, s, "Oro", 20, int(rng.integers(0, 6))) for s in (CHAMPION, CHALLENGER) for _ in range(30)]
    r = bootstrap_differences(build_window(pd.DataFrame(rows)), seed=42).iloc[0]
    assert r["ic_inferior"] <= r["diferencia"] <= r["ic_superior"]
