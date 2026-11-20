"""Statistical transformations for alluvial plots: stat_stratum, stat_alluvium, stat_flow."""

from __future__ import annotations

import pandas as pd
from plotnine.stats.stat import stat

from ._layout import compute_alluvia_panel, compute_flows_panel, compute_strata_panel


class stat_stratum(stat):
    """Aggregate and stack strata at each alluvial axis."""

    REQUIRED_AES: set[str] = set()
    DEFAULT_PARAMS = {
        "geom": "stratum",
        "position": "identity",
        "na_rm": False,
        "decreasing": None,
        "reverse": True,
        "discern": False,
    }
    CREATES = {"x", "y", "ymin", "ymax", "stratum", "count", "prop", "n", "deposit"}

    def compute_panel(self, data: pd.DataFrame, scales) -> pd.DataFrame:
        return compute_strata_panel(
            data,
            decreasing=self.params.get("decreasing", None),
            reverse=bool(self.params.get("reverse", True)),
        )


class stat_alluvium(stat):
    """Compute stacked lode positions for continuous alluvia across axes."""

    REQUIRED_AES: set[str] = set()
    DEFAULT_PARAMS = {
        "geom": "alluvium",
        "position": "identity",
        "na_rm": False,
        "decreasing": None,
        "reverse": True,
        "lode_guidance": "zigzag",
        "aes_bind": "none",
        "discern": False,
    }
    CREATES = {"x", "y", "ymin", "ymax", "stratum", "alluvium"}

    def compute_panel(self, data: pd.DataFrame, scales) -> pd.DataFrame:
        return compute_alluvia_panel(
            data,
            decreasing=self.params.get("decreasing", None),
            reverse=bool(self.params.get("reverse", True)),
            lode_guidance=self.params.get("lode_guidance", "zigzag") or "zigzag",
            aes_bind=self.params.get("aes_bind", "none") or "none",
        )


class stat_flow(stat):
    """Compute stacked pairwise flows between adjacent alluvial axes."""

    REQUIRED_AES: set[str] = set()
    DEFAULT_PARAMS = {
        "geom": "flow",
        "position": "identity",
        "na_rm": False,
        "decreasing": None,
        "reverse": True,
        "aes_bind": "none",
        "aes_flow": "forward",
        "discern": False,
    }
    CREATES = {"x", "y", "ymin", "ymax", "stratum", "alluvium", "flow"}

    def compute_panel(self, data: pd.DataFrame, scales) -> pd.DataFrame:
        return compute_flows_panel(
            data,
            decreasing=self.params.get("decreasing", None),
            reverse=bool(self.params.get("reverse", True)),
            aes_bind=self.params.get("aes_bind", "none") or "none",
            aes_flow=self.params.get("aes_flow", "forward") or "forward",
        )
