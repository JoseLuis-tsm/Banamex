"""Pruebas de scripts/run_fase.py: nombres de fase, orden de scripts y fase vacía."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_fase", ROOT / "scripts" / "run_fase.py")
run_fase = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_fase)


@pytest.mark.parametrize(
    "fase, esperado",
    [("1", "01"), ("01", "01"), ("5a", "05a"), ("5B", "05b"), ("05b", "05b"), ("6", "06")],
)
def test_normalize_phase(fase, esperado):
    assert run_fase.normalize_phase(fase) == esperado


@pytest.mark.parametrize("fase", ["x", "12", "5c", ""])
def test_normalize_phase_invalida(fase):
    with pytest.raises(ValueError):
        run_fase.normalize_phase(fase)


def test_find_phase_dir_existe_para_cada_fase():
    for fase in ["1", "2", "3", "4", "5a", "5b", "6"]:
        assert run_fase.find_phase_dir(fase) is not None


def test_list_phase_scripts_orden_desde_y_plantilla(tmp_path):
    for name in ["02_b.py", "01_a.py", "_plantilla.py", "10_c.py", "notas.md", "3_mal.py"]:
        (tmp_path / name).write_text("", encoding="utf-8")
    nombres = [p.name for p in run_fase.list_phase_scripts(tmp_path)]
    assert nombres == ["01_a.py", "02_b.py", "10_c.py"]
    desde = [p.name for p in run_fase.list_phase_scripts(tmp_path, start="02")]
    assert desde == ["02_b.py", "10_c.py"]


def test_fase_vacia_avisa_y_sale_con_cero():
    # La fase 6 no tendrá scripts hasta el cierre; --desde 99 deja cualquier fase vacía.
    result = subprocess.run(
        [sys.executable, "scripts/run_fase.py", "1", "--desde", "99"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0
    assert "no tiene scripts" in result.stdout


def test_fase_inexistente_sale_con_error():
    result = subprocess.run(
        [sys.executable, "scripts/run_fase.py", "9"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
