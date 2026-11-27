"""Pruebas de madurez y de los conjuntos de modelado (src/data/cohorts.py)."""

import pandas as pd
import pytest

from src.data import catalog as cat
from src.data.cohorts import (
    S1_TARGET, S3_TARGET, build_s1_dataset, build_s3_dataset, is_mature, mature_vintages,
    maturity_table, over30_curves, over30_exceeds_enr,
)


def _todas_las_vintages():
    return [y * 100 + m for y in (2024, 2025, 2026) for m in range(1, 13) if y * 100 + m <= 202609]


def test_las_vintages_maduras_a_6_son_las_27_esperadas():
    maduras = mature_vintages(_todas_las_vintages(), horizon=6)
    assert len(maduras) == 27
    assert maduras[0] == 202401 and maduras[-1] == 202603
    assert not is_mature(202604, 6) and is_mature(202603, 6)


def _cohorte(vintage, strategy, mobs, enr=4, cuentas=5, linea=200000, o30=None):
    filas = []
    for mob in mobs:
        filas.append({
            "VINTAGE": vintage, "STRATEGY": strategy, "STRATEGY_SEGMENT": "Costco", "PRODUCT": "Costco",
            "CHANNEL": "Costco", "AGE_TAG": "03| Mature >30", "SCORE_TAG": "151-160", "MOB": mob,
            "ACCOUNTS": cuentas, "CREDITLINESAMTS": linea, "ENRACCTS": enr,
            "OVER30ACCTS": (o30 if o30 is not None else min(mob, 2)),
        })
    return filas


@pytest.fixture
def base():
    filas = (
        _cohorte(202601, "r.Reca champ", range(9))           # madura
        + _cohorte(202604, "r.Reca champ", range(6))         # no madura (MOB máx 5)
        + _cohorte(202608, "DARIA 2.0", range(2), linea=50000)  # no evaluable en riesgo
        + _cohorte(202602, "DARIA 1.6", range(8), enr=0)     # ENRACCTS = 0 en MOB 6
    )
    return pd.DataFrame(filas)


def test_s1_toma_mob_0_de_todas_las_vintages(base):
    s1 = build_s1_dataset(base)
    assert len(s1) == 4
    assert set(s1["VINTAGE"]) == {202601, 202604, 202608, 202602}
    assert s1.loc[s1["VINTAGE"] == 202601, S1_TARGET].iloc[0] == pytest.approx(200000 / 5)
    assert "MOB" not in s1.columns


def test_s3_solo_maduras_evaluables_y_con_saldo(base):
    s3 = build_s3_dataset(base, horizon=6)
    assert list(s3["VINTAGE"]) == [202601]
    assert s3[S3_TARGET].iloc[0] == pytest.approx(2 / 4)
    assert "MOB" not in s3.columns


def test_s3_no_incluye_daria_2_aunque_tuviera_mob_6():
    df = pd.DataFrame(_cohorte(202601, "DARIA 2.0", range(7)))
    assert build_s3_dataset(df, horizon=6).empty


def test_curvas_y_proporcion_over30_mayor_enr(base):
    curvas = over30_curves(base[base["VINTAGE"] == 202601], ["VINTAGE"])
    assert curvas.loc[curvas["MOB"] == 6, "tasa_over30_enr"].iloc[0] == pytest.approx(2 / 4)
    assert curvas.loc[curvas["MOB"] == 6, "tasa_over30_cuentas"].iloc[0] == pytest.approx(2 / 5)

    df = pd.DataFrame(_cohorte(202601, "r.Reca champ", range(7), enr=1, o30=3) + _cohorte(202602, "r.Reca champ", range(7)))
    df.loc[df["VINTAGE"] == 202602, "STRATEGY_SEGMENT"] = "Brands-Payroll-OnUs"
    t = over30_exceeds_enr(df, mob=6)
    assert t["cohortes"].iloc[0] == 2 and t["proporcion"].iloc[0] == pytest.approx(0.5)


def test_tabla_de_madurez(base):
    t = maturity_table(base, horizon=6).set_index("VINTAGE")
    assert t.loc[202601, "madura"] and not t.loc[202604, "madura"]
    assert t.loc[202608, "mob_max_posible"] == cat.max_mob(202608) == 1
