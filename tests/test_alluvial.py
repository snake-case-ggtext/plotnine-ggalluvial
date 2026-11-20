"""Unit tests for plotnine-ggalluvial."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from plotnine import aes, after_stat, coord_flip, facet_wrap, geom_label, geom_text, ggplot

from plotnine_ggalluvial import (
    geom_alluvium,
    geom_flow,
    geom_stratum,
    stat_alluvium,
    stat_flow,
    stat_stratum,
)
from plotnine_ggalluvial._curves import build_alluvium_polygon, build_flow_polygon, flow_curve_boundary
from plotnine_ggalluvial._layout import lode_guidance_order


def _sample_wide() -> pd.DataFrame:
    return pd.DataFrame({
        "axis1": ["A", "A", "B", "B"],
        "axis2": ["X", "Y", "X", "Y"],
        "axis3": ["P", "Q", "Q", "P"],
        "freq": [10, 20, 15, 5],
        "grp": ["G1", "G1", "G2", "G2"],
        "fac": ["F1", "F1", "F2", "F2"],
    })


def _sample_lodes() -> pd.DataFrame:
    return pd.DataFrame({
        "time": ["T1", "T2", "T3", "T1", "T2", "T3"],
        "state": ["Low", "Mid", "High", "High", "Low", "Mid"],
        "subject": ["S1", "S1", "S1", "S2", "S2", "S2"],
        "freq": [12, 12, 12, 8, 8, 8],
        "grp": ["A", "A", "A", "B", "B", "B"],
    })


def _draw(p):
    fig = p.draw()
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    return fig


@pytest.mark.parametrize("curve_type", ["xspline", "linear", "cubic", "quintic", "sine", "arctangent", "sigmoid"])
def test_all_curve_types_monotone_and_endpoints(curve_type):
    xs, ys = flow_curve_boundary(1.0, 2.0, 10.0, 40.0, curve_type=curve_type)
    assert np.isclose(xs[0], 1.0) and np.isclose(xs[-1], 2.0)
    assert np.isclose(ys[0], 10.0) and np.isclose(ys[-1], 40.0)
    assert np.all(np.diff(xs) >= -1e-9)
    assert np.all(np.diff(ys) >= -1e-9)


def test_xspline_matches_blanc_schlick_midpoint():
    xs, ys = flow_curve_boundary(0.0, 1.0, 0.0, 1.0, knot_pos=0.25, knot_prop=True, curve_type="xspline")
    ymid = float(np.interp(0.5, xs, ys))
    assert abs(ymid - 0.5) < 1e-4
    y2 = float(np.interp(0.2, xs, ys))
    assert abs(y2 - 0.07188) < 5e-4


def test_lode_guidance_orders():
    assert lode_guidance_order(3, 1, "zigzag") == [1, 2, 0]
    assert lode_guidance_order(3, 1, "zagzig") == [1, 0, 2]
    assert lode_guidance_order(4, 1, "zigzag") == [1, 0, 2, 3]
    assert lode_guidance_order(4, 2, "zigzag") == [2, 3, 1, 0]
    assert lode_guidance_order(4, 1, "frontback") == [1, 2, 3, 0]
    assert lode_guidance_order(4, 2, "backfront") == [2, 1, 0, 3]
    assert lode_guidance_order(3, 2, "forward") == [2, 0, 1]
    assert lode_guidance_order(3, 0, "backward") == [0, 2, 1]


def test_wide_alluvium_stratum_and_labels_render():
    df = _sample_wide()
    p = (
        ggplot(df, aes(y="freq", axis1="axis1", axis2="axis2", axis3="axis3"))
        + geom_alluvium(aes(fill="grp"))
        + geom_stratum()
        + geom_text(aes(label=after_stat("stratum")), stat="stratum")
    )
    fig = _draw(p)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].collections) >= 2


def test_lodes_flow_backward_and_geom_label_render():
    df = _sample_lodes()
    p = (
        ggplot(df, aes(x="time", stratum="state", alluvium="subject", y="freq"))
        + geom_flow(aes(fill="state"), aes_flow="backward", curve_type="cubic")
        + geom_stratum(fill="#f3f4f6", color="#111827")
        + geom_label(aes(label=after_stat("stratum")), stat="stratum")
    )
    fig = _draw(p)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].collections) >= 2


def test_decreasing_reverse_facets_and_coord_flip():
    df = _sample_wide()
    for dec in (True, False, None):
        p = (
            ggplot(df, aes(y="freq", axis1="axis1", axis2="axis2"))
            + geom_alluvium(aes(fill="grp"), decreasing=dec, reverse=False)
            + geom_stratum(decreasing=dec, reverse=False)
            + facet_wrap("~fac")
        )
        fig = _draw(p)
        assert len(fig.axes) == 2

    p_flip = (
        ggplot(df, aes(y="freq", axis1="axis1", axis2="axis2"))
        + geom_alluvium(aes(fill="grp"), curve_type="sigmoid")
        + geom_stratum()
        + coord_flip()
    )
    fig_flip = _draw(p_flip)
    assert len(fig_flip.axes) == 1


def test_polygon_builders_and_exports():
    fx, fy = build_flow_polygon(1.16, 1.84, 0.0, 10.0, 5.0, 15.0, curve_type="quintic")
    assert len(fx) == len(fy) >= 10
    ax, ay = build_alluvium_polygon(
        np.array([0.83, 1.83]), np.array([1.17, 2.17]),
        np.array([0.0, 10.0]), np.array([5.0, 15.0]),
        curve_type="xspline",
    )
    assert len(ax) == len(ay) >= 10
    assert stat_alluvium is not None and stat_stratum is not None and stat_flow is not None
