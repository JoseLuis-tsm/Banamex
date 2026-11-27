"""Pruebas de S4 (src/strategy/s4.py): ventana, soporte común y bootstrap."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.s4 import (
    CHALLENGER, CHAMPION, NO_CONCLUSIVE, NO_EVIDENCE, STATUS_COMMON, STATUS_NO_EVIDENCE, bootstrap_differences, build_window,
    decide, leave_one_vintage_out, profile_table, sign_agreement,
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


# ===== Método 2 =====
def test_eventos_esperados_dan_la_tasa_predicha_ponderada(s3):
    w = build_window(s3)
    w["E"] = w["ENRACCTS"] * np.where(w["STRATEGY"] == CHALLENGER, 0.3, 0.1)
    oro = bootstrap_differences(w, seed=42, n_boot=100, events_column="E").set_index("PERFIL").loc["Oro | Branch | 03| Mature >30"]
    assert oro["diferencia"] == pytest.approx(0.2)


def test_dejar_fuera_una_vintage_no_entrena_con_ella(monkeypatch):
    import xgboost as xgb

    from src.features.engineering import build_features
    from src.models import tuning

    base = {"STRATEGY": CHAMPION, "STRATEGY_SEGMENT": "Costco", "PRODUCT": "Costco", "CHANNEL": "Costco",
            "AGE_TAG": "01| Young < 25", "SCORE_TAG": "<130"}
    rows = pd.DataFrame([{**base, "VINTAGE": v} for v in (202510, 202511, 202512)])
    frame = pd.concat([rows, build_features(rows)], axis=1).assign(TASA_OVER30=0.1, ENRACCTS=10)
    seen = []

    def fake_train(params, dtrain, num_rounds):
        seen.append(dtrain.num_row())
        return xgb.train({"objective": "binary:logistic"}, dtrain, num_boost_round=1)

    monkeypatch.setattr(tuning, "train_booster", fake_train)
    window = rows[rows["VINTAGE"] >= 202511].reset_index(drop=True)
    pred = leave_one_vintage_out(frame, window, {}, 1, 1.0, 0.0)
    assert seen == [2, 2]  # en cada vuelta entrena con las otras dos vintages
    assert pred.notna().all() and pred.between(0, 1).all()


# ===== Regla de decisión =====
def _m1(rows):
    return pd.DataFrame(rows, columns=["PERFIL", "diferencia", "ic_inferior", "ic_superior", "estimable", "concluyente"])


def test_regla_de_decision_cubre_todos_los_casos():
    m1 = _m1([
        ("a", 0.05, 0.01, 0.09, True, True),     # concluyente, mismo signo → champ
        ("b", -0.04, -0.07, -0.01, True, True),  # concluyente, mismo signo → DARIA
        ("c", 0.05, 0.01, 0.09, True, True),     # concluyente, signo opuesto → sin diferencia
        ("d", 0.02, -0.01, 0.05, True, False),   # intervalo incluye 0
        ("e", 0.03, np.nan, np.nan, False, False),  # no estimable
    ])
    m2 = pd.DataFrame({"PERFIL": list("abcde"), "diferencia": [0.03, -0.02, -0.01, 0.02, 0.03]})
    profiles = pd.DataFrame({"PERFIL": list("abcdef"), "estado": [STATUS_COMMON] * 5 + [STATUS_NO_EVIDENCE]})
    d = decide(m1, m2, profiles).set_index("PERFIL")
    assert d.loc["a", "sugerencia"] == CHAMPION and d.loc["b", "sugerencia"] == CHALLENGER
    assert (d.loc[list("cde"), "sugerencia"] == NO_CONCLUSIVE).all()
    assert d.loc["f", "sugerencia"] == NO_EVIDENCE
    assert "signo" in d.loc["c", "motivo"] and "incluye cero" in d.loc["d", "motivo"] and "no estimable" in d.loc["e", "motivo"]
    agr = sign_agreement(d.reset_index())
    assert agr["perfiles_concluyentes"] == 3 and agr["coinciden"] == 2 and agr["concordancia"] == pytest.approx(2 / 3)


def test_diferencia_cero_en_metodo_2_no_cuenta_como_coincidencia():
    m1 = _m1([("a", 0.05, 0.01, 0.09, True, True)])
    d = decide(m1, pd.DataFrame({"PERFIL": ["a"], "diferencia": [0.0]}), pd.DataFrame({"PERFIL": ["a"], "estado": [STATUS_COMMON]}))
    assert d.loc[0, "sugerencia"] == NO_CONCLUSIVE
