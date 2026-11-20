"""Alluvial data normalization, stratum ordering, lode guidance, and vertical stacking."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

_AXIS_RE = re.compile(r"^axis(\d+)$")
_DIFF_AES = ("fill", "color", "colour", "alpha", "linetype", "linewidth", "size")


def _axis_cols(df: pd.DataFrame) -> list[str]:
    matches = []
    for col in df.columns:
        m = _AXIS_RE.match(str(col))
        if m:
            matches.append((int(m.group(1)), col))
    matches.sort(key=lambda t: t[0])
    return [c for _, c in matches]


def normalize_to_lodes(data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Convert either wide (axis1, axis2, ...) or long (x, stratum, alluvium) data into lodes form.

    Returns (lodes_df, is_wide).
    """
    df = data.copy()
    axes = _axis_cols(df)

    if axes:
        # Wide (alluvia) format -> long lodes format
        if "y" not in df.columns:
            df["y"] = 1.0
        else:
            df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)

        keep_cols = [c for c in df.columns if c not in axes and c not in ("x", "stratum", "alluvium")]
        n_rows = len(df)
        frames = []
        for k, col in enumerate(axes, start=1):
            part = df[keep_cols].copy()
            part["x"] = float(k)
            part["stratum"] = df[col].astype(str)
            part["alluvium"] = np.arange(1, n_rows + 1, dtype=int)
            frames.append(part)
        out = pd.concat(frames, ignore_index=True)
        return out, True

    # Long (lodes) format
    if "x" not in df.columns:
        raise ValueError("Alluvial stats require either `axis1, axis2, ...` or `x` with `stratum`/`alluvium`.")
    if "y" not in df.columns:
        df["y"] = 1.0
    else:
        df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    if "stratum" not in df.columns and "alluvium" in df.columns:
        df["stratum"] = df["alluvium"]
    if "alluvium" not in df.columns and "stratum" in df.columns:
        df["alluvium"] = np.arange(1, len(df) + 1, dtype=int)
    df["stratum"] = df["stratum"].astype(str)
    return df, False


def _stratum_level_order(series: pd.Series) -> list[str]:
    """Determine canonical category order (Categorical order if categorical, else sorted strings)."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        cats = [str(c) for c in series.cat.categories]
        present = set(series.astype(str))
        return [c for c in cats if c in present]
    return sorted(series.astype(str).unique().tolist())


def _order_strata_at_axis(
    totals: pd.Series,
    level_order: list[str],
    decreasing: bool | None,
    reverse: bool,
) -> list[str]:
    """Return stratum names ordered from bottom to top of the vertical stack."""
    rank_map = {lvl: i for i, lvl in enumerate(level_order)}
    items = list(totals.index)

    if decreasing is None:
        # Order by variable values: reverse=True puts first level on TOP (last in bottom-to-top list)
        items.sort(key=lambda s: rank_map.get(s, 0), reverse=bool(reverse))
        return items

    if decreasing:
        # Largest on bottom -> descending total from bottom to top
        items.sort(key=lambda s: (-float(totals[s]), -rank_map.get(s, 0) if reverse else rank_map.get(s, 0)))
        return items

    # decreasing is False: largest on top -> ascending total from bottom to top
    items.sort(key=lambda s: (float(totals[s]), -rank_map.get(s, 0) if reverse else rank_map.get(s, 0)))
    return items


def lode_guidance_order(n: int, i: int, guidance: str = "zigzag") -> list[int]:
    """Return 0-based axis priority order of length n starting at 0-based index i."""
    g = (guidance or "zigzag").lower()
    if g in ("forward", "rightward"):
        return [i] + [k for k in range(n) if k != i]
    if g in ("backward", "leftward"):
        return [i] + [k for k in range(n - 1, -1, -1) if k != i]
    if g in ("frontback", "rightleft"):
        return [i] + list(range(i + 1, n)) + list(range(i - 1, -1, -1))
    if g in ("backfront", "leftright"):
        return [i] + list(range(i - 1, -1, -1)) + list(range(i + 1, n))

    # zigzag / zagzig: alternate outward from i
    left = list(range(i - 1, -1, -1))
    right = list(range(i + 1, n))
    # Outward direction is toward the nearer edge (right on exact middle tie)
    outward_left = i < (n - 1) / 2.0
    if g == "zagzig":
        outward_left = not outward_left

    first, second = (left, right) if outward_left else (right, left)
    order = [i]
    for a, b in zip(first, second):
        order.extend([a, b])
    if len(first) > len(second):
        order.extend(first[len(second):])
    elif len(second) > len(first):
        order.extend(second[len(first):])
    return order


def _aes_sort_cols(df: pd.DataFrame) -> list[str]:
    """Differentiation aesthetic columns present in df with >1 unique value."""
    cols = []
    for c in _DIFF_AES:
        if c in df.columns and df[c].nunique(dropna=False) > 1:
            cols.append(c)
    return cols


def _aes_key_val(val: Any, reverse: bool) -> tuple:
    s = "" if pd.isna(val) else str(val)
    return (s,)


def compute_strata_panel(
    data: pd.DataFrame,
    *,
    decreasing: bool | None = None,
    reverse: bool = True,
) -> pd.DataFrame:
    """Compute stratum rectangles (x, y, ymin, ymax, stratum, count, prop, n, deposit) for one panel."""
    lodes, _ = normalize_to_lodes(data)
    if lodes.empty:
        return pd.DataFrame()

    panel_val = lodes["PANEL"].iloc[0] if "PANEL" in lodes.columns else 1
    xs = sorted(lodes["x"].unique().tolist())

    rows = []
    deposit_counter = 0
    for xv in xs:
        sub = lodes[lodes["x"] == xv]
        lvl_order = _stratum_level_order(sub["stratum"])
        grouped = sub.groupby("stratum", sort=False)["y"].agg(["sum", "count"])
        totals = grouped["sum"]
        ordered_strata = _order_strata_at_axis(totals, lvl_order, decreasing=decreasing, reverse=reverse)
        axis_total = float(totals.sum())

        y_cursor = 0.0
        for s_name in ordered_strata:
            deposit_counter += 1
            h = float(totals[s_name])
            n_cases = int(grouped.loc[s_name, "count"])
            ymin = y_cursor
            ymax = y_cursor + h
            y_cursor = ymax
            rows.append({
                "PANEL": panel_val,
                "x": xv,
                "stratum": s_name,
                "y": 0.5 * (ymin + ymax),
                "ymin": ymin,
                "ymax": ymax,
                "count": h,
                "n": n_cases,
                "prop": h / axis_total if axis_total > 0 else 0.0,
                "deposit": deposit_counter,
                "group": deposit_counter,
            })

    return pd.DataFrame(rows)


def compute_alluvia_panel(
    data: pd.DataFrame,
    *,
    decreasing: bool | None = None,
    reverse: bool = True,
    lode_guidance: str = "zigzag",
    aes_bind: str = "none",
) -> pd.DataFrame:
    """Compute stacked lode positions for continuous alluvia across axes in one panel."""
    lodes, _ = normalize_to_lodes(data)
    if lodes.empty:
        return pd.DataFrame()

    panel_val = lodes["PANEL"].iloc[0] if "PANEL" in lodes.columns else 1
    xs = sorted(lodes["x"].unique().tolist())
    n_axes = len(xs)

    # Compute per-axis stratum deposit rank (0 = bottom, 1 = next, ...)
    deposit_by_axis: dict[Any, dict[str, int]] = {}
    for xv in xs:
        sub = lodes[lodes["x"] == xv]
        lvl_order = _stratum_level_order(sub["stratum"])
        totals = sub.groupby("stratum", sort=False)["y"].sum()
        ordered = _order_strata_at_axis(totals, lvl_order, decreasing=decreasing, reverse=reverse)
        deposit_by_axis[xv] = {s: idx for idx, s in enumerate(ordered)}

    # Map each alluvium -> its stratum deposit rank at every axis
    alluv_ids = list(dict.fromkeys(lodes["alluvium"].tolist()))
    alluv_to_idx = {aid: idx for idx, aid in enumerate(alluv_ids)}
    dep_matrix: dict[Any, dict[Any, int]] = {aid: {} for aid in alluv_ids}
    for row in lodes[["alluvium", "x", "stratum"]].itertuples(index=False):
        dep_matrix[row.alluvium][row.x] = deposit_by_axis[row.x].get(row.stratum, 0)

    bind = (aes_bind or "none").lower()
    aes_cols = _aes_sort_cols(lodes)

    # Precompute rank maps for aesthetic columns (respecting reverse for categorical consistency)
    aes_rank_maps: dict[str, dict[str, int]] = {}
    for col in aes_cols:
        lvls = _stratum_level_order(lodes[col].astype(str))
        if reverse:
            lvls = list(reversed(lvls))
        aes_rank_maps[col] = {v: idx for idx, v in enumerate(lvls)}

    out_frames = []
    for axis_i, xv in enumerate(xs):
        sub = lodes[lodes["x"] == xv].copy()
        prio = lode_guidance_order(n_axes, axis_i, guidance=lode_guidance)
        prio_xs = [xs[idx] for idx in prio]

        sort_keys = []
        for idx_row, r in sub.iterrows():
            aid = r["alluvium"]
            deps = [dep_matrix[aid].get(px, -1) for px in prio_xs]
            a_ranks = [aes_rank_maps[c].get(str(r[c]), 0) for c in aes_cols]
            tie = -alluv_to_idx[aid] if reverse else alluv_to_idx[aid]
            if bind == "alluvia":
                key = (deps[0], tuple(a_ranks), tuple(deps[1:]), tie)
            elif bind == "flows":
                key = (tuple(deps), tuple(a_ranks), tie)
            else:
                key = (tuple(deps), tie)
            sort_keys.append((key, idx_row))

        sort_keys.sort(key=lambda item: item[0])
        ordered_indices = [idx_row for _, idx_row in sort_keys]
        sub = sub.loc[ordered_indices].copy()

        ys = sub["y"].to_numpy(dtype=float)
        ymaxs = np.cumsum(ys)
        ymins = ymaxs - ys
        sub["ymin"] = ymins
        sub["ymax"] = ymaxs
        sub["y"] = 0.5 * (ymins + ymaxs)
        sub["group"] = sub["alluvium"]
        sub["PANEL"] = panel_val
        out_frames.append(sub)

    return pd.concat(out_frames, ignore_index=True)


def compute_flows_panel(
    data: pd.DataFrame,
    *,
    decreasing: bool | None = None,
    reverse: bool = True,
    aes_bind: str = "none",
    aes_flow: str = "forward",
) -> pd.DataFrame:
    """Compute pairwise adjacent-axis flows (re-stacked independently at each gap)."""
    lodes, _ = normalize_to_lodes(data)
    if lodes.empty:
        return pd.DataFrame()

    panel_val = lodes["PANEL"].iloc[0] if "PANEL" in lodes.columns else 1
    xs = sorted(lodes["x"].unique().tolist())
    if len(xs) < 2:
        return pd.DataFrame()

    # Per-axis stratum deposit rank (0 = bottom)
    deposit_by_axis: dict[Any, dict[str, int]] = {}
    for xv in xs:
        sub = lodes[lodes["x"] == xv]
        lvl_order = _stratum_level_order(sub["stratum"])
        totals = sub.groupby("stratum", sort=False)["y"].sum()
        ordered = _order_strata_at_axis(totals, lvl_order, decreasing=decreasing, reverse=reverse)
        deposit_by_axis[xv] = {s: idx for idx, s in enumerate(ordered)}

    bind = (aes_bind or "none").lower()
    flow_dir = (aes_flow or "forward").lower()
    aes_cols = [c for c in _DIFF_AES if c in lodes.columns]

    aes_rank_maps: dict[str, dict[str, int]] = {}
    for col in aes_cols:
        lvls = _stratum_level_order(lodes[col].astype(str))
        if reverse:
            lvls = list(reversed(lvls))
        aes_rank_maps[col] = {v: idx for idx, v in enumerate(lvls)}

    rows_out = []
    flow_id = 0

    for k in range(len(xs) - 1):
        x0, x1 = xs[k], xs[k + 1]
        left = lodes[lodes["x"] == x0].set_index("alluvium")
        right = lodes[lodes["x"] == x1].set_index("alluvium")
        common = [aid for aid in left.index if aid in right.index]
        if not common:
            continue

        # Build pairwise flow records
        pair_recs = []
        for aid in common:
            lr = left.loc[aid]
            rr = right.loc[aid]
            if isinstance(lr, pd.DataFrame):
                lr = lr.iloc[0]
            if isinstance(rr, pd.DataFrame):
                rr = rr.iloc[0]
            s_from = str(lr["stratum"])
            s_to = str(rr["stratum"])
            src = rr if flow_dir == "backward" else lr
            rec: dict[str, Any] = {
                "s_from": s_from,
                "s_to": s_to,
                "y": float(lr["y"]),
                "dep_from": deposit_by_axis[x0].get(s_from, 0),
                "dep_to": deposit_by_axis[x1].get(s_to, 0),
            }
            for c in aes_cols:
                rec[c] = src[c]
            pair_recs.append(rec)

        pdf = pd.DataFrame(pair_recs)

        # Aggregate flows sharing (s_from, s_to) and active differentiation aesthetics
        group_cols = ["s_from", "s_to", "dep_from", "dep_to"] + aes_cols
        agg = pdf.groupby(group_cols, sort=False, dropna=False)["y"].sum().reset_index()

        # Assign unique flow_id to each aggregated flow
        agg["_fid"] = np.arange(flow_id + 1, flow_id + 1 + len(agg), dtype=int)
        flow_id += len(agg)

        # Compute stacking at x0 (from) and x1 (to):
        # Differentiating aesthetics break ties within identical (dep_from, dep_to) flows
        # (and are placed before the adjacent axis when aes_bind == "alluvia").
        def _sort_key_from(r):
            a_r = tuple(aes_rank_maps[c].get(str(r[c]), 0) for c in aes_cols)
            if bind == "alluvia":
                return (int(r["dep_from"]), a_r, int(r["dep_to"]), int(r["_fid"]))
            return (int(r["dep_from"]), int(r["dep_to"]), a_r, int(r["_fid"]))

        def _sort_key_to(r):
            a_r = tuple(aes_rank_maps[c].get(str(r[c]), 0) for c in aes_cols)
            if bind == "alluvia":
                return (int(r["dep_to"]), a_r, int(r["dep_from"]), int(r["_fid"]))
            return (int(r["dep_to"]), int(r["dep_from"]), a_r, int(r["_fid"]))

        from_order = sorted(range(len(agg)), key=lambda i: _sort_key_from(agg.iloc[i]))
        to_order = sorted(range(len(agg)), key=lambda i: _sort_key_to(agg.iloc[i]))

        from_bounds: dict[int, tuple[float, float]] = {}
        cur = 0.0
        for i in from_order:
            h = float(agg.iloc[i]["y"])
            from_bounds[i] = (cur, cur + h)
            cur += h

        to_bounds: dict[int, tuple[float, float]] = {}
        cur = 0.0
        for i in to_order:
            h = float(agg.iloc[i]["y"])
            to_bounds[i] = (cur, cur + h)
            cur += h

        # Emit two rows per flow (from and to) sharing group = _fid
        for i in range(len(agg)):
            r = agg.iloc[i]
            fid = int(r["_fid"])
            ymin0, ymax0 = from_bounds[i]
            ymin1, ymax1 = to_bounds[i]
            base_aes = {c: r[c] for c in aes_cols}
            rows_out.append({
                "PANEL": panel_val, "group": fid, "alluvium": fid, "flow": "from",
                "x": x0, "stratum": r["s_from"],
                "y": 0.5 * (ymin0 + ymax0), "ymin": ymin0, "ymax": ymax0,
                **base_aes,
            })
            rows_out.append({
                "PANEL": panel_val, "group": fid, "alluvium": fid, "flow": "to",
                "x": x1, "stratum": r["s_to"],
                "y": 0.5 * (ymin1 + ymax1), "ymin": ymin1, "ymax": ymax1,
                **base_aes,
            })

    return pd.DataFrame(rows_out)
