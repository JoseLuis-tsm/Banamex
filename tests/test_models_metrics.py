"""Pruebas de métricas ponderadas, líneas base, registro y calibración (src/models/)."""

import numpy as np
import pandas as pd
import pytest

from src.models import registry
from src.models.baselines import RatioBaseline
from src.models.metrics import (
    calibration_by_decile, is_increasing, metrics_table, rate_by_risk_level, rate_metrics, regression_metrics,
    risk_level, risk_thresholds, score_monotonic_violations, weighted_quantiles,
)
from src.models.predict import logit, platt_transform, sigmoid


# ===== S1 =====
def test_metricas_de_regresion_ponderadas():
    m = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]), np.array([1.0, 3.0]))
    assert m["mae"] == pytest.approx((10 + 3 * 20) / 4)
    assert m["mape_pct"] == pytest.approx(100 * (0.1 + 3 * 0.1) / 4)
    assert m["sesgo"] == pytest.approx((10 - 60) / 4)
    assert m["sesgo_pct"] == pytest.approx(100 * ((110 + 540) / 700 - 1))
    assert regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 1.0]))["r2"] == 1.0


# ===== S3 =====
def test_metricas_de_tasa_equivalen_a_registros_expandidos():
    from sklearn.metrics import log_loss, roc_auc_score

    e, n, p = np.array([1, 3, 0]), np.array([4, 5, 2]), np.array([0.2, 0.5, 0.1])
    m = rate_metrics(e, n, p)
    # Expansión explícita: cada cuenta es un registro
    y = np.concatenate([[1] * 1 + [0] * 3, [1] * 3 + [0] * 2, [0] * 2])
    s = np.concatenate([[0.2] * 4, [0.5] * 5, [0.1] * 2])
    assert m["auc"] == pytest.approx(roc_auc_score(y, s))
    assert m["log_loss"] == pytest.approx(log_loss(y, s))
    assert m["brier"] == pytest.approx(np.mean((y - s) ** 2))
    assert m["obs_pred"] == pytest.approx(4 / (0.8 + 2.5 + 0.2))
    assert m["gini"] == pytest.approx(2 * m["auc"] - 1)


def test_metricas_de_tasa_sin_eventos_no_fallan():
    m = rate_metrics(np.array([0, 0]), np.array([3, 4]), np.array([0.1, 0.2]))
    assert np.isnan(m["auc"]) and np.isnan(m["ks"]) and m["log_loss"] > 0


def test_cuantiles_ponderados_y_niveles_de_riesgo():
    v, w = np.array([0.1, 0.2, 0.3, 0.4]), np.array([1, 1, 1, 1])
    assert list(weighted_quantiles(v, w, [0.5, 0.75])) == [0.2, 0.3]
    t = risk_thresholds(np.arange(1, 101) / 100, np.ones(100))
    assert t == {"p50": 0.5, "p80": 0.8, "p95": 0.95}
    assert list(risk_level(np.array([0.5, 0.51, 0.8, 0.95, 0.99]), t)) == ["Bajo", "Medio", "Medio", "Alto", "Crítico"]


def test_calibracion_por_deciles_y_por_nivel():
    rng = np.random.default_rng(42)
    p = rng.uniform(0.01, 0.3, 500)
    n = rng.integers(1, 20, 500).astype(float)
    e = np.round(n * p)
    cal = calibration_by_decile(e, n, p)
    assert len(cal) == 10 and cal["cuentas_expuestas"].sum() == n.sum()
    assert is_increasing(cal["tasa_predicha"].tolist())
    lv = rate_by_risk_level(e, n, p, risk_thresholds(p, n))
    assert list(lv["nivel"]) == ["Bajo", "Medio", "Alto", "Crítico"]
    assert is_increasing(lv["tasa_observada"].tolist())


def test_violaciones_monotonas():
    df = pd.DataFrame({"g": ["a"] * 3 + ["b"] * 3, "s": [0, 0.5, 1] * 2, "p": [0.3, 0.2, 0.1, 0.1, 0.2, 0.05]})
    assert score_monotonic_violations(df, "p", "s", ["g"]) == 1


def test_metrics_table_desglosa_por_conjunto_y_grupo():
    df = pd.DataFrame({"CONJUNTO": ["validacion"] * 2 + ["prueba"] * 2, "G": ["x", "y", "x", "y"],
                       "y": [1.0, 2.0, 3.0, 4.0], "p": [1.0, 2.0, 3.0, 5.0], "w": [1.0] * 4})
    t = metrics_table(df, "m", regression_metrics, ["validacion", "prueba"], ["G"], y="y", pred="p", w="w")
    assert len(t) == 6
    assert t[(t.conjunto == "prueba") & (t.grupo == "y")]["mae"].iloc[0] == 1.0


# ===== Líneas base =====
def test_linea_base_razon_de_sumas_con_respaldo():
    train = pd.DataFrame({"P": ["a", "a", "b"], "S": ["1", "2", "1"], "num": [10.0, 30.0, 8.0], "den": [1.0, 3.0, 2.0]})
    base = RatioBaseline([["P", "S"], ["P"], []], "num", "den").fit(train)
    new = pd.DataFrame({"P": ["a", "a", "c"], "S": ["1", "9", "1"]})
    pred = base.predict(new)
    assert pred[0] == pytest.approx(10.0)          # grupo exacto
    assert pred[1] == pytest.approx(40.0 / 4.0)     # solo producto
    assert pred[2] == pytest.approx(48.0 / 6.0)     # global
    assert list(base.last_levels) == [0, 1, 2]


# ===== Calibración =====
def test_platt_identidad_y_monotonia():
    p = np.array([0.01, 0.1, 0.5, 0.9])
    assert platt_transform(p, 1.0, 0.0) == pytest.approx(p)
    assert sigmoid(logit(p)) == pytest.approx(p)
    cal = platt_transform(p, 0.8, -0.3)
    assert is_increasing(cal.tolist())


# ===== Registro =====
def test_registro_guarda_publica_y_carga(tmp_path):
    v = registry.version_id("s9", {"a": 1})
    assert v == registry.version_id("s9", {"a": 1}) and v != registry.version_id("s9", {"a": 2})
    registry.save_metadata(tmp_path, "s9", v, {"metricas": {"mae": 1.0}})
    assert registry.current(tmp_path, "s9") is None
    registry.publish(tmp_path, "s9", v)
    folder, meta = registry.current(tmp_path, "s9")
    assert folder == tmp_path / "s9" / v and meta["version"] == v and "fecha" in meta and "python" in meta["versiones_librerias"]


def test_sha256_de_archivo(tmp_path):
    f = tmp_path / "x.csv"
    f.write_bytes(b"a,b\n1,2\n")  # bytes exactos (write_text cambia \n por \r\n en Windows)
    import hashlib
    assert registry.file_sha256(f) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
