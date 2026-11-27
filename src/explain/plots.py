"""Gráficas SHAP con Plotly: barras de importancia, beeswarm y cascada de un perfil.

Todas las vistas llevan la nota fija de no causalidad y la de escala (add_notes()).
"""

from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.explain.shap_values import NOTE, SCALE_NOTE, VARIABLE_LABELS, VARIABLES
from src.utils.charts import CATEGORICAL, TEXT_SECONDARY, style

UP_COLOR = CATEGORICAL[7]    # rojo: la contribución sube la tasa estimada
DOWN_COLOR = CATEGORICAL[0]  # azul: la contribución baja la tasa estimada

# Variables numéricas en el modelo (el color del beeswarm muestra su valor).
NUMERIC_VALUES = {"AGE_TAG": "EDAD_ORDINAL", "SCORE_TAG": "SCORE_POSICION", "BOOKING_MONTH": "MES_BOOKING"}


def add_notes(fig: go.Figure) -> go.Figure:
    """Agrega al pie de la gráfica la nota de no causalidad y la de escala."""
    fig.add_annotation(text=NOTE + "<br>" + SCALE_NOTE, xref="paper", yref="paper", x=0, y=-0.22,
                       showarrow=False, align="left", font={"size": 11, "color": TEXT_SECONDARY})
    fig.update_layout(margin={"b": 140})
    return fig


def importance_bars(importance: pd.DataFrame, title: str) -> go.Figure:
    """Barras horizontales: media |SHAP| por variable original (mayor arriba)."""
    t = importance.iloc[::-1]
    fig = go.Figure(go.Bar(x=t["importancia"], y=t["etiqueta"], orientation="h", marker_color=CATEGORICAL[0], name="Importancia",
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(showlegend=False, height=480)
    return add_notes(style(fig, title, "Media de |SHAP| (escala logit)", ""))


def beeswarm(grouped: pd.DataFrame, features: pd.DataFrame, values: pd.DataFrame, title: str,
             order: List[str], max_points: int = 3000, seed: int = 42) -> go.Figure:
    """Beeswarm: un punto por cohorte y variable; x = contribución SHAP, y = variable (con dispersión).

    Para edad, score y mes el color muestra el valor de la variable (claro = bajo, oscuro =
    alto); para las categóricas el color es neutro y el valor aparece al pasar el cursor.
    """
    rng = np.random.default_rng(seed)
    idx = grouped.index if len(grouped) <= max_points else rng.choice(grouped.index, max_points, replace=False)
    fig = go.Figure()
    for i, var in enumerate(order[::-1]):
        jitter = rng.uniform(-0.3, 0.3, len(idx))
        x = grouped.loc[idx, var]
        if var in NUMERIC_VALUES:
            raw = features.loc[idx, NUMERIC_VALUES[var]]
            span = raw.max() - raw.min()
            color = (raw - raw.min()) / span if span > 0 else raw * 0
            marker = {"size": 5, "color": color, "colorscale": [[0, "#cfe0f6"], [1, "#0d3f80"]], "opacity": 0.7}
        else:
            marker = {"size": 5, "color": "#8a8984", "opacity": 0.45}
        fig.add_scatter(x=x, y=i + jitter, mode="markers", marker=marker, name=VARIABLE_LABELS[var], showlegend=False,
                        customdata=values.loc[idx, var], hovertemplate=VARIABLE_LABELS[var] + ": %{customdata}<br>SHAP %{x:+.3f}<extra></extra>")
    fig.add_vline(x=0, line_color=TEXT_SECONDARY, line_width=1)
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=[VARIABLE_LABELS[v] for v in order[::-1]])
    fig.update_layout(height=520)
    return add_notes(style(fig, title, "Contribución SHAP (escala logit; positiva = sube la tasa estimada)", ""))


def waterfall(contrib: pd.Series, base: float, output: float, labels: pd.Series, title: str) -> go.Figure:
    """Cascada de un perfil: valor base → contribución de cada variable → salida (logit)."""
    order = contrib.abs().sort_values(ascending=False).index
    text = [f"{VARIABLE_LABELS[v]} = {labels[v]}" for v in order]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute"] + ["relative"] * len(order) + ["total"],
        x=["Valor base"] + text + ["Salida del modelo"], y=[base] + [contrib[v] for v in order] + [output],
        increasing={"marker": {"color": UP_COLOR}}, decreasing={"marker": {"color": DOWN_COLOR}},
        totals={"marker": {"color": "#8a8984"}}, connector={"line": {"color": "#d8d7d2"}},
        texttemplate="%{delta:+.3f}", hovertemplate="%{x}<br>%{y:+.4f}<extra></extra>", name="SHAP",
    ))
    fig.update_layout(showlegend=False, height=560)
    fig.update_xaxes(tickangle=-30)
    return add_notes(style(fig, title, "", "Logit de la tasa sin calibrar"))


__all__ = ["add_notes", "beeswarm", "importance_bars", "waterfall", "VARIABLES"]
