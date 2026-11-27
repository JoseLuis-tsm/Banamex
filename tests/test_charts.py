"""Pruebas del estilo común de gráficas (src/utils/charts.py)."""

import plotly.graph_objects as go
import pytest

from src.data.catalog import STRATEGIES
from src.utils.charts import CATEGORICAL, STRATEGY_COLORS, color_map, save_html, style


def test_cada_estrategia_tiene_un_color_distinto_y_fijo():
    assert set(STRATEGY_COLORS) == set(STRATEGIES)
    assert len(set(STRATEGY_COLORS.values())) == len(STRATEGIES)


def test_color_map_respeta_el_orden_y_el_limite():
    assert color_map(["a", "b"]) == {"a": CATEGORICAL[0], "b": CATEGORICAL[1]}
    with pytest.raises(ValueError):
        color_map([str(i) for i in range(9)])


def test_style_y_save_html(tmp_path):
    fig = style(go.Figure(go.Bar(x=[1], y=[2])), "Título", "x", "y")
    assert fig.layout.title.text == "Título"
    out = save_html(fig, tmp_path / "sub" / "g.html")
    assert out.exists() and "plotly" in out.read_text(encoding="utf-8").lower()
