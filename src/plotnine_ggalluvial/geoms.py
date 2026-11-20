"""Geometric objects for alluvial plots: geom_stratum, geom_alluvium, geom_flow."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from plotnine.geoms.geom import geom
from plotnine.geoms.geom_polygon import geom_polygon

from ._curves import build_alluvium_polygon, build_flow_polygon


def _rgba(color: Any, alpha: Any) -> tuple[float, float, float, float]:
    if color is None or (isinstance(color, float) and np.isnan(color)):
        return (0.0, 0.0, 0.0, 0.0)
    s = str(color).strip().lower()
    if s in ("", "none", "transparent", "na", "nan"):
        return (0.0, 0.0, 0.0, 0.0)
    a = 1.0 if (alpha is None or (isinstance(alpha, float) and np.isnan(alpha))) else float(alpha)
    r, g, b, base_a = to_rgba(s)
    return (r, g, b, base_a * a)


def _edge_rgba(color: Any) -> tuple[float, float, float, float]:
    if color is None or (isinstance(color, float) and np.isnan(color)):
        return (0.0, 0.0, 0.0, 0.0)
    s = str(color).strip().lower()
    if s in ("", "none", "transparent", "na", "nan"):
        return (0.0, 0.0, 0.0, 0.0)
    return to_rgba(s)


class geom_stratum(geom):
    """Draw axis-aligned stratum rectangles at each alluvial axis."""

    DEFAULT_AES = {
        "alpha": 1.0,
        "color": "black",
        "fill": "white",
        "linetype": "solid",
        "size": 0.5,
    }
    REQUIRED_AES = {"x", "ymin", "ymax"}
    DEFAULT_PARAMS = {
        "stat": "stratum",
        "position": "identity",
        "na_rm": False,
        "width": 1.0 / 3.0,
    }
    legend_geom = "polygon"
    draw_legend = staticmethod(geom_polygon.draw_legend)

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data
        data = data.copy()
        w = float(self.params.get("width", 1.0 / 3.0))
        data["xmin"] = data["x"].astype(float) - 0.5 * w
        data["xmax"] = data["x"].astype(float) + 0.5 * w
        return data

    def draw_panel(self, data: pd.DataFrame, panel_params, coord, ax):
        if data.empty:
            return
        params = self.params
        verts = []
        facecolors = []
        edgecolors = []
        linewidths = []
        linestyles = []

        for _, row in data.iterrows():
            poly_df = pd.DataFrame({
                "x": [row["xmin"], row["xmax"], row["xmax"], row["xmin"]],
                "y": [row["ymin"], row["ymin"], row["ymax"], row["ymax"]],
            })
            tdf = coord.transform(poly_df, panel_params)
            verts.append(np.column_stack([tdf["x"].to_numpy(float), tdf["y"].to_numpy(float)]))
            facecolors.append(_rgba(row.get("fill", "white"), row.get("alpha", 1.0)))
            ec = _edge_rgba(row.get("color", "black"))
            edgecolors.append(ec)
            lw = float(row.get("size", 0.5)) * (72.0 / 25.4) * 0.5
            linewidths.append(lw if ec[3] > 0 else 0.0)
            linestyles.append(row.get("linetype", "solid") or "solid")

        coll = PolyCollection(
            verts,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=linewidths,
            linestyles=linestyles,
            zorder=params.get("zorder", 2),
            rasterized=params.get("raster", False),
        )
        ax.add_collection(coll)


class geom_alluvium(geom):
    """Draw continuous curved alluvial ribbons spanning all axes."""

    DEFAULT_AES = {
        "alpha": 0.5,
        "color": "none",
        "fill": "#bebebe",
        "linetype": "solid",
        "size": 0.5,
    }
    REQUIRED_AES = {"x", "ymin", "ymax"}
    DEFAULT_PARAMS = {
        "stat": "alluvium",
        "position": "identity",
        "na_rm": False,
        "width": 1.0 / 3.0,
        "knot_pos": 0.25,
        "knot_prop": True,
        "curve_type": None,
        "curve_range": None,
        "segments": None,
    }
    legend_geom = "polygon"
    draw_legend = staticmethod(geom_polygon.draw_legend)

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data
        data = data.copy()
        w = float(self.params.get("width", 1.0 / 3.0))
        data["xmin"] = data["x"].astype(float) - 0.5 * w
        data["xmax"] = data["x"].astype(float) + 0.5 * w
        return data

    def draw_panel(self, data: pd.DataFrame, panel_params, coord, ax):
        if data.empty:
            return
        params = self.params
        knot_pos = float(params.get("knot_pos", 0.25))
        knot_prop = bool(params.get("knot_prop", True))
        curve_type = params.get("curve_type", None)
        curve_range = params.get("curve_range", None)
        segments = params.get("segments", None)

        verts = []
        facecolors = []
        edgecolors = []
        linewidths = []
        linestyles = []

        for _, grp in data.groupby("group", sort=False):
            g = grp.sort_values("x")
            if len(g) < 2:
                continue
            xs, ys = build_alluvium_polygon(
                g["xmin"].to_numpy(float),
                g["xmax"].to_numpy(float),
                g["ymin"].to_numpy(float),
                g["ymax"].to_numpy(float),
                knot_pos=knot_pos,
                knot_prop=knot_prop,
                curve_type=curve_type,
                curve_range=curve_range,
                segments=segments,
            )
            tdf = coord.transform(pd.DataFrame({"x": xs, "y": ys}), panel_params)
            verts.append(np.column_stack([tdf["x"].to_numpy(float), tdf["y"].to_numpy(float)]))
            row0 = g.iloc[0]
            alpha = row0.get("alpha", 0.5)
            facecolors.append(_rgba(row0.get("fill", "#bebebe"), alpha))
            ec = _rgba(row0.get("color", "none"), alpha)
            edgecolors.append(ec)
            lw = float(row0.get("size", 0.5)) * (72.0 / 25.4) * 0.5
            linewidths.append(lw if ec[3] > 0 else 0.0)
            linestyles.append(row0.get("linetype", "solid") or "solid")

        if verts:
            coll = PolyCollection(
                verts,
                facecolors=facecolors,
                edgecolors=edgecolors,
                linewidths=linewidths,
                linestyles=linestyles,
                zorder=params.get("zorder", 2),
                rasterized=params.get("raster", False),
            )
            ax.add_collection(coll)


class geom_flow(geom):
    """Draw curved pairwise flow ribbons between adjacent strata."""

    DEFAULT_AES = {
        "alpha": 0.5,
        "color": "none",
        "fill": "#bebebe",
        "linetype": "solid",
        "size": 0.5,
    }
    REQUIRED_AES = {"x", "ymin", "ymax"}
    DEFAULT_PARAMS = {
        "stat": "flow",
        "position": "identity",
        "na_rm": False,
        "width": 1.0 / 3.0,
        "knot_pos": 0.25,
        "knot_prop": True,
        "curve_type": None,
        "curve_range": None,
        "segments": None,
        "aes_flow": "forward",
    }
    legend_geom = "polygon"
    draw_legend = staticmethod(geom_polygon.draw_legend)

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data
        data = data.copy()
        w = float(self.params.get("width", 1.0 / 3.0))
        data["xmin"] = data["x"].astype(float) - 0.5 * w
        data["xmax"] = data["x"].astype(float) + 0.5 * w
        return data

    def draw_panel(self, data: pd.DataFrame, panel_params, coord, ax):
        if data.empty:
            return
        params = self.params
        knot_pos = float(params.get("knot_pos", 0.25))
        knot_prop = bool(params.get("knot_prop", True))
        curve_type = params.get("curve_type", None)
        curve_range = params.get("curve_range", None)
        segments = params.get("segments", None)
        aes_flow = (params.get("aes_flow", "forward") or "forward").lower()

        verts = []
        facecolors = []
        edgecolors = []
        linewidths = []
        linestyles = []

        for _, grp in data.groupby("group", sort=False):
            g = grp.sort_values("x")
            if len(g) < 2:
                continue
            r0, r1 = g.iloc[0], g.iloc[-1]
            xs, ys = build_flow_polygon(
                float(r0["xmax"]),
                float(r1["xmin"]),
                float(r0["ymin"]),
                float(r0["ymax"]),
                float(r1["ymin"]),
                float(r1["ymax"]),
                knot_pos=knot_pos,
                knot_prop=knot_prop,
                curve_type=curve_type,
                curve_range=curve_range,
                segments=segments,
            )
            tdf = coord.transform(pd.DataFrame({"x": xs, "y": ys}), panel_params)
            verts.append(np.column_stack([tdf["x"].to_numpy(float), tdf["y"].to_numpy(float)]))
            src = r1 if aes_flow == "backward" else r0
            alpha = src.get("alpha", 0.5)
            facecolors.append(_rgba(src.get("fill", "#bebebe"), alpha))
            ec = _rgba(src.get("color", "none"), alpha)
            edgecolors.append(ec)
            lw = float(src.get("size", 0.5)) * (72.0 / 25.4) * 0.5
            linewidths.append(lw if ec[3] > 0 else 0.0)
            linestyles.append(src.get("linetype", "solid") or "solid")

        if verts:
            coll = PolyCollection(
                verts,
                facecolors=facecolors,
                edgecolors=edgecolors,
                linewidths=linewidths,
                linestyles=linestyles,
                zorder=params.get("zorder", 2),
                rasterized=params.get("raster", False),
            )
            ax.add_collection(coll)


# Make geom_polygon import explicitly used only for legend compatibility if needed
_ = geom_polygon
