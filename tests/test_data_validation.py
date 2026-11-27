"""Pruebas del validador del contrato (src/data/validation.py) con casos rotos a propósito."""

import numpy as np
import pandas as pd
import pytest

from src.data.validation import BLOCKING, WARNING, cumulative_evidence, has_blocking_failures, validate


def _valida() -> pd.DataFrame:
    """Dos cohortes válidas de la vintage 202603 (MOB 0 a 6), una por familia de score."""
    filas = []
    for segmento, score, cuentas in (("Costco", "151-160", 10), ("Brands-Payroll-OnUs", "<130", 6)):
        for mob in range(7):
            gcl, rec = 100.0 * mob, 40.0 * mob
            acq = 50.0 if mob == 0 else 0.0
            filas.append({
                "VINTAGE": 202603, "STRATEGY": "r.Reca champ", "STRATEGY_SEGMENT": segmento,
                "PRODUCT": "Costco", "CHANNEL": "Costco", "AGE_TAG": "02| Young 25-30", "SCORE_TAG": score,
                "MOB": mob, "ACCOUNTS": cuentas, "CREDITLINESAMTS": cuentas * 40000, "ENRACCTS": cuentas - 1,
                "ENRAMTS": 1000.0 + mob, "OVER30ACCTS": min(mob, 3), "OVER30AMTS": 10.0 * min(mob, 3),
                "GCLAMTS": gcl, "NCLAMTS": gcl - rec, "RECOVERYAMTS": rec, "REVENUE": 5.0 * mob,
                "EXPENSES_ACQ": acq, "EXPENSES_ECM_VAR": 20.0, "EXPENSES_ECM_FIX": 10.0,
                "EXPENSES_TOTAL": acq + 30.0,
            })
    return pd.DataFrame(filas)


def _estado(reporte: pd.DataFrame, regla: str) -> pd.Series:
    return reporte.set_index("regla").loc[regla]


def test_datos_validos_pasan_todas_las_reglas():
    reporte = validate(_valida())
    assert not has_blocking_failures(reporte)
    assert (reporte["filas_afectadas"] == 0).all()


def _romper(cambio):
    df = _valida()
    cambio(df)
    return df


@pytest.mark.parametrize(
    "regla, cambio",
    [
        ("llave_repetida", lambda df: df.__setitem__("MOB", df["MOB"].where(df.index != 1, 0))),
        ("valor_nulo", lambda df: df.__setitem__("REVENUE", df["REVENUE"].where(df.index != 3, np.nan))),
        ("categoria_fuera_de_catalogo:PRODUCT", lambda df: df.__setitem__("PRODUCT", "Inventado")),
        ("categoria_fuera_de_catalogo:STRATEGY", lambda df: df.__setitem__("STRATEGY", "DARIA 9")),
        ("categoria_fuera_de_catalogo:SCORE_TAG", lambda df: df.__setitem__("SCORE_TAG", "a.<=154")),  # score DARIA en Recalibration
        ("categoria_fuera_de_catalogo:STRATEGY_SEGMENT", lambda df: df.__setitem__("STRATEGY_SEGMENT", "LE Thin")),
        ("estrategia_no_corresponde_a_vintage", lambda df: df.__setitem__("STRATEGY", "15.Recalibration 2.0")),
        ("mob_mayor_al_posible", lambda df: df.__setitem__("MOB", df["MOB"] + 1)),
        ("valor_fuera_de_rango:ACCOUNTS", lambda df: df.__setitem__("ACCOUNTS", -1)),
        ("valor_fuera_de_rango:ENRACCTS", lambda df: df.__setitem__("ENRACCTS", 2.5)),  # no entero
        ("valor_fuera_de_rango:VINTAGE", lambda df: df.__setitem__("VINTAGE", 202312)),
        ("ncl_distinto_de_gcl_menos_recuperacion", lambda df: df.__setitem__("NCLAMTS", df["NCLAMTS"] + 1)),
        ("expenses_total_distinto_de_la_suma", lambda df: df.__setitem__("EXPENSES_TOTAL", df["EXPENSES_TOTAL"] + 0.01)),
        ("expenses_acq_con_mob_distinto_de_0", lambda df: df.__setitem__("EXPENSES_ACQ", 5.0)),
    ],
)
def test_regla_bloqueante_detecta_el_caso_roto(regla, cambio):
    reporte = validate(_romper(cambio))
    fila = _estado(reporte, regla)
    assert fila["severidad"] == BLOCKING
    assert fila["estado"] == "FALLA" and fila["filas_afectadas"] > 0
    assert fila["que_hacer"]
    assert has_blocking_failures(reporte)


def test_columna_faltante_bloquea_y_detiene_las_demas_reglas():
    reporte = validate(_valida().drop(columns=["GCLAMTS"]))
    assert has_blocking_failures(reporte)
    assert list(reporte["regla"]) == ["columna_faltante", "columna_no_declarada"]


@pytest.mark.parametrize(
    "regla, cambio",
    [
        ("over30_mayor_que_enr", lambda df: df.__setitem__("OVER30ACCTS", df["ENRACCTS"] + 1)),
        ("acumulada_baja:REVENUE", lambda df: df.__setitem__("REVENUE", df["REVENUE"].where(df["MOB"] != 4, 0.0))),
        ("columna_no_declarada", lambda df: df.__setitem__("OTRA", 1)),
    ],
)
def test_los_avisos_no_bloquean(regla, cambio):
    reporte = validate(_romper(cambio))
    fila = _estado(reporte, regla)
    assert fila["severidad"] == WARNING and fila["estado"] == "avisa"
    assert not has_blocking_failures(reporte)


def test_el_reporte_da_la_linea_del_csv():
    df = _valida()
    df.loc[3, "PRODUCT"] = "Inventado"
    fila = _estado(validate(df), "categoria_fuera_de_catalogo:PRODUCT")
    assert fila["ejemplo_filas"] == "5"  # posición 3 + encabezado + base 1


def test_evidencia_de_columnas_acumuladas():
    tabla = cumulative_evidence(_valida(), ["ACCOUNTS", "REVENUE", "EXPENSES_TOTAL"]).set_index("columna")
    assert tabla.loc["ACCOUNTS", "conclusion"] == "constante en la cohorte"
    assert tabla.loc["REVENUE", "conclusion"] == "acumulada"
    assert tabla.loc["EXPENSES_TOTAL", "conclusion"] == "del mes (no acumulada)"


def test_la_evidencia_no_mezcla_cohortes():
    # Si las filas vienen desordenadas, la diferencia debe seguir calculándose dentro de cada cohorte.
    df = _valida().sample(frac=1, random_state=42)
    tabla = cumulative_evidence(df, ["REVENUE"]).set_index("columna")
    assert tabla.loc["REVENUE", "pasos_que_bajan"] == 0


def test_el_script_sale_con_error_ante_fallas_bloqueantes(tmp_path):
    import subprocess
    import sys

    from src.utils import PROJECT_ROOT

    script = PROJECT_ROOT / "scripts" / "01_datos" / "02_validacion_contrato.py"
    for nombre, df, codigo in (("valida", _valida(), 0), ("rota", _romper(lambda d: d.__setitem__("PRODUCT", "X")), 1)):
        entrada = tmp_path / f"{nombre}.csv"
        df.to_csv(entrada, index=False, encoding="utf-8")
        salida = tmp_path / nombre
        r = subprocess.run(
            [sys.executable, str(script), "--entrada", str(entrada), "--dir-reportes", str(salida)],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert r.returncode == codigo, r.stdout + r.stderr
        assert (salida / "validacion.csv").exists()
