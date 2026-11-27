"""Estilo común de las gráficas Plotly de los reportes (HTML con fig.write_html()).

Paleta categórica en orden fijo (paleta de referencia validada para daltonismo en
pares adyacentes). El color sigue a la entidad, no a su rango: cada estrategia y cada
familia tiene siempre el mismo color en todas las gráficas.
"""

from pathlib import Path
from typing import Dict, List, Union

import plotly.graph_objects as go

# Paleta categórica (modo claro), en el orden en que se asignan los colores.
CATEGORICAL: List[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Rampa secuencial de un solo tono (azul), de claro a oscuro, para magnitudes (mapas de calor).
SEQUENTIAL_BLUE: List[List[Union[float, str]]] = [[0.0, "#eef4fc"], [0.5, "#7fb0ea"], [1.0, "#0d3f80"]]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"

STRATEGY_COLORS: Dict[str, str] = {
    "14.Recalibration 1.0": CATEGORICAL[0],
    "15.Recalibration 2.0": CATEGORICAL[1],
    "r.Reca champ": CATEGORICAL[2],
    "DARIA 1.6": CATEGORICAL[3],
    "DARIA 2.0": CATEGORICAL[4],
}
FAMILY_COLORS: Dict[str, str] = {"Recalibration": CATEGORICAL[0], "DARIA": CATEGORICAL[1]}


def color_map(values: List[str]) -> Dict[str, str]:
    """Asigna colores de la paleta en orden fijo a una lista de valores (máximo 8)."""
    if len(values) > len(CATEGORICAL):
        raise ValueError(f"Hay {len(values)} series; el máximo es {len(CATEGORICAL)}. Agrupa en 'Otros' o separa la gráfica.")
    return {v: CATEGORICAL[i] for i, v in enumerate(values)}


def style(fig: go.Figure, title: str, x_title: str = "", y_title: str = "") -> go.Figure:
    """Aplica el estilo común: fondo claro, rejilla tenue, títulos en español y hover unificado."""
    fig.update_layout(
        title={"text": title, "x": 0.01, "font": {"size": 18, "color": TEXT_PRIMARY}},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "Segoe UI, Helvetica, Arial, sans-serif", "size": 13, "color": TEXT_SECONDARY},
        legend={"title": {"text": ""}, "orientation": "h", "y": -0.18, "x": 0},
        margin={"l": 70, "r": 30, "t": 70, "b": 90},
        bargap=0.25,
        hoverlabel={"bgcolor": "white", "font": {"color": TEXT_PRIMARY}},
    )
    fig.update_xaxes(title_text=x_title, showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_traces(selector={"type": "bar"}, marker_line_width=1, marker_line_color=SURFACE)
    fig.update_traces(selector={"type": "scatter"}, line_width=2)
    return fig


def save_html(fig: go.Figure, path: Path) -> Path:
    """Guarda la figura como HTML autocontenido (plotly.js incrustado, funciona sin internet)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    return path
