"""Pruebas de las reglas de consistencia (src/consistency/rules.py), una o más por regla."""

import numpy as np
import pandas as pd
import pytest

from src.consistency.rules import (
    ALERT_TEXT, STATUS_NOT_RISK, STATUS_OK, STATUS_OUT_S1, STATUS_OUT_S3, apply_rules, check_rules, confidence_level,
    domain_keys, in_domain, similar_accounts, similar_accounts_table,
)

THRESHOLDS = {"p50": 0.05, "p80": 0.10, "p95": 0.15}
VERSIONS = {"modelo_s1": "s1-x", "modelo_s3": "s3-x", "datos_sha256": "abc", "horizonte_mob": 6}


def _p(strategy="r.Reca champ", seg="Costco", product="Costco", channel="Costco", age="01| Young < 25", score="<130", vintage=202601):
    return {"VINTAGE": vintage, "STRATEGY": strategy, "STRATEGY_SEGMENT": seg, "PRODUCT": product, "CHANNEL": channel,
            "AGE_TAG": age, "SCORE_TAG": score}


@pytest.fixture
def training():
    rows = [dict(_p(), ACCOUNTS=800), dict(_p(score="130-140"), ACCOUNTS=300), dict(_p(age="03| Mature >30"), ACCOUNTS=150)]
    return pd.DataFrame(rows)


def test_dominio_usa_familia_y_no_la_estrategia_ni_edad_ni_score(training):
    keys = domain_keys(training)
    same_family = pd.DataFrame([_p(strategy="14.Recalibration 1.0", age="02| Young 25-30", score="141-150")])
    other_channel = pd.DataFrame([_p(channel="Internet")])
    assert in_domain(same_family, keys).all() and not in_domain(other_channel, keys).any()


def test_perfiles_similares_suman_cualquier_score(training):
    table = similar_accounts_table(training)
    acc = similar_accounts(pd.DataFrame([_p(score="171-180"), _p(age="03| Mature >30"), _p(channel="Internet")]), table)
    assert list(acc) == [1100, 150, 0]


def test_niveles_de_confianza_en_los_bordes():
    assert list(confidence_level([1000, 999, 200, 199, 0])) == ["Alta", "Media", "Media", "Baja", "Baja"]


def _run(training, profiles, line, rate, planned=None, domain_s3=None):
    keys = domain_keys(training)
    out = apply_rules(pd.DataFrame(profiles), np.array(line, float), np.array(rate, float), THRESHOLDS, keys,
                      keys if domain_s3 is None else domain_s3, similar_accounts_table(training), VERSIONS,
                      planned_accounts=None if planned is None else np.array(planned, float))
    return out


def test_s2_se_calcula_y_la_tasa_se_recorta(training):
    out = _run(training, [_p(), _p()], [10_000, 20_000], [1.3, -0.2], planned=[3, 0])
    assert list(out["monto_total"]) == [30_000, 0]
    assert list(out["tasa_impago"]) == [1.0, 0.0]


def test_daria_2_no_es_evaluable_en_riesgo(training):
    t = pd.DataFrame([dict(_p(strategy="DARIA 1.6", seg="LE Thin", score="i.205+"), ACCOUNTS=10)])
    tr = pd.concat([training, t], ignore_index=True)
    out = _run(tr, [_p(strategy="DARIA 2.0", seg="LE Thin", score="i.205+")], [5_000], [0.2])
    assert out.loc[0, "estado"] == STATUS_NOT_RISK and np.isnan(out.loc[0, "tasa_impago"]) and out.loc[0, "motivo"]
    assert out.loc[0, "linea_promedio"] == 5_000  # la línea sí se da: DARIA 2.0 tiene MOB 0
    # Sin tasa no hay confianza que reportar; las cuentas similares quedan como dato informativo.
    assert out.loc[0, "confianza"] == "No aplica" and out.loc[0, "cuentas_similares"] > 0


def test_fuera_de_dominio_no_recibe_la_salida(training):
    out = _run(training, [_p(channel="Internet")], [5_000], [0.1], planned=[2])
    assert out.loc[0, "estado"] == STATUS_OUT_S1
    assert out[["linea_promedio", "monto_total", "tasa_impago"]].isna().all(axis=None)
    out3 = _run(training, [_p()], [5_000], [0.1], domain_s3=set())
    assert out3.loc[0, "estado"] == STATUS_OUT_S3 and np.isnan(out3.loc[0, "tasa_impago"]) and out3.loc[0, "linea_promedio"] == 5_000


def test_alerta_solo_con_riesgo_critico(training):
    out = _run(training, [_p()] * 4, [1] * 4, [0.04, 0.08, 0.12, 0.20])
    assert list(out["nivel_riesgo"]) == ["Bajo", "Medio", "Alto", "Crítico"]
    assert list(out["alerta"]) == ["", "", "", ALERT_TEXT]


def test_versiones_y_estado_evaluable(training):
    out = _run(training, [_p()], [1], [0.05])
    assert out.loc[0, "estado"] == STATUS_OK and out.loc[0, "motivo"] == ""
    assert out.loc[0, "modelo_s3"] == "s3-x" and out.loc[0, "horizonte_mob"] == 6


def test_check_rules_pasa_y_detecta_una_falla(training):
    profiles = [_p(), _p(channel="Internet"), _p(strategy="DARIA 2.0", seg="LE Thin", score="i.205+")]
    line, rate = [10, 20, 30], [0.2, 0.1, 0.1]
    out = _run(training, profiles, line, rate, planned=[1, 1, 1])
    assert check_rules(out, np.array(rate), np.array(line), THRESHOLDS)["cumple_100"].all()
    broken = out.copy()
    broken.loc[0, "alerta"] = ""
    t = check_rules(broken, np.array(rate), np.array(line), THRESHOLDS).set_index("regla")
    assert not t.loc["alerta si y solo si Crítico", "cumple_100"]
