"""Curve generators for alluvial flows and ribbons."""

from __future__ import annotations

import math

import numpy as np


def _eval_unit_curve(
    t: np.ndarray,
    curve_type: str,
    curve_range: float | None = None,
) -> np.ndarray:
    """Evaluate a monotone bijection f: [0, 1] -> [0, 1] for non-xspline curve types."""
    ct = (curve_type or "xspline").lower()
    if ct == "linear":
        return t
    if ct == "cubic":
        return 3.0 * t**2 - 2.0 * t**3
    if ct == "quintic":
        return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    if ct == "sine":
        return 0.5 * (1.0 - np.cos(math.pi * t))
    if ct in ("arctangent", "arctan"):
        r = (2.0 + math.sqrt(3.0)) if curve_range is None else float(curve_range)
        g = np.arctan(r * (2.0 * t - 1.0))
        g0 = math.atan(-r)
        g1 = math.atan(r)
        return (g - g0) / max(g1 - g0, 1e-12)
    if ct == "sigmoid":
        r = 6.0 if curve_range is None else float(curve_range)
        g = 1.0 / (1.0 + np.exp(-r * (2.0 * t - 1.0)))
        g0 = 1.0 / (1.0 + math.exp(r))
        g1 = 1.0 / (1.0 + math.exp(-r))
        return (g - g0) / max(g1 - g0, 1e-12)
    raise ValueError(f"Unknown curve_type: {curve_type!r}")


def _xspline_blend(u: np.ndarray, p: float) -> np.ndarray:
    """Blanc & Schlick (1995) Eq. (11) approximating X-spline blending function."""
    return (u**3) * (10.0 - p + (2.0 * p - 15.0) * u + (6.0 - p) * (u**2))


def _eval_xspline_4pt(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    n_per_seg: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a 4-control-point approximating X-spline with shapes (0, 1, 1, 0).

    Implements Blanc & Schlick (1995) approximating X-splines where support width
    is (1 + s) and p = 2 * (1 + s)^2.
    """
    pts = np.array([p0, p1, p2, p3], dtype=float)
    shapes = [0.0, 1.0, 1.0, 0.0]

    xs_out: list[np.ndarray] = []
    ys_out: list[np.ndarray] = []

    for k in range(3):
        pk_prev = pts[k - 1] if k > 0 else pts[0]
        pk = pts[k]
        pk_next = pts[k + 1]
        pk_next2 = pts[k + 2] if (k + 2) < 4 else pts[3]

        s1 = shapes[k]
        s2 = shapes[k + 1]
        p_s1 = 2.0 * (1.0 + s1) ** 2
        p_s2 = 2.0 * (1.0 + s2) ** 2

        t = np.linspace(0.0, 1.0, n_per_seg, endpoint=(k == 2))

        if s1 > 0.0:
            u0 = np.clip((s1 - t) / (1.0 + s1), 0.0, 1.0)
            a0 = np.where(t < s1, _xspline_blend(u0, p_s1), 0.0)
        else:
            a0 = np.zeros_like(t)

        u1 = np.clip((1.0 + s2 - t) / (1.0 + s2), 0.0, 1.0)
        a1 = _xspline_blend(u1, p_s2)

        u2 = np.clip((t + s1) / (1.0 + s1), 0.0, 1.0)
        a2 = _xspline_blend(u2, p_s1)

        if s2 > 0.0:
            u3 = np.clip((t - (1.0 - s2)) / (1.0 + s2), 0.0, 1.0)
            a3 = np.where(t > 1.0 - s2, _xspline_blend(u3, p_s2), 0.0)
        else:
            a3 = np.zeros_like(t)

        wsum = a0 + a1 + a2 + a3
        cx = (a0 * pk_prev[0] + a1 * pk[0] + a2 * pk_next[0] + a3 * pk_next2[0]) / wsum
        cy = (a0 * pk_prev[1] + a1 * pk[1] + a2 * pk_next[1] + a3 * pk_next2[1]) / wsum
        xs_out.append(cx)
        ys_out.append(cy)

    return np.concatenate(xs_out), np.concatenate(ys_out)


def flow_curve_boundary(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    knot_pos: float = 0.25,
    knot_prop: bool = True,
    curve_type: str | None = None,
    curve_range: float | None = None,
    segments: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (x, y) points along a single boundary curve from (x0, y0) to (x1, y1)."""
    ct = (curve_type or "xspline").lower()
    if ct in ("xspline", "spline"):
        gap = x1 - x0
        kp = knot_pos * gap if knot_prop else knot_pos
        p0 = (x0, y0)
        p1 = (x0 + kp, y0)
        p2 = (x1 - kp, y1)
        p3 = (x1, y1)
        return _eval_xspline_4pt(p0, p1, p2, p3, n_per_seg=20)

    n_seg = 48 if segments is None else max(3, int(segments))
    t = np.linspace(0.0, 1.0, n_seg + 1)
    xs = x0 + (x1 - x0) * t
    ys = y0 + (y1 - y0) * _eval_unit_curve(t, ct, curve_range=curve_range)
    return xs, ys


def build_flow_polygon(
    x0: float,
    x1: float,
    ymin0: float,
    ymax0: float,
    ymin1: float,
    ymax1: float,
    *,
    knot_pos: float = 0.25,
    knot_prop: bool = True,
    curve_type: str | None = None,
    curve_range: float | None = None,
    segments: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build closed polygon (x, y) for a single inter-stratum flow ribbon."""
    xt, yt = flow_curve_boundary(
        x0, x1, ymax0, ymax1,
        knot_pos=knot_pos, knot_prop=knot_prop,
        curve_type=curve_type, curve_range=curve_range, segments=segments,
    )
    xb, yb = flow_curve_boundary(
        x0, x1, ymin0, ymin1,
        knot_pos=knot_pos, knot_prop=knot_prop,
        curve_type=curve_type, curve_range=curve_range, segments=segments,
    )
    xs = np.concatenate([xt, xb[::-1]])
    ys = np.concatenate([yt, yb[::-1]])
    return xs, ys


def build_alluvium_polygon(
    xmins: np.ndarray,
    xmaxs: np.ndarray,
    ymins: np.ndarray,
    ymaxs: np.ndarray,
    *,
    knot_pos: float = 0.25,
    knot_prop: bool = True,
    curve_type: str | None = None,
    curve_range: float | None = None,
    segments: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build closed polygon (x, y) for a continuous multi-axis alluvium ribbon."""
    n = len(xmins)
    top_x: list[np.ndarray] = []
    top_y: list[np.ndarray] = []
    bot_x: list[np.ndarray] = []
    bot_y: list[np.ndarray] = []

    for i in range(n):
        # Flat lode segment across axis i
        top_x.append(np.array([xmins[i], xmaxs[i]], dtype=float))
        top_y.append(np.array([ymaxs[i], ymaxs[i]], dtype=float))
        bot_x.append(np.array([xmins[i], xmaxs[i]], dtype=float))
        bot_y.append(np.array([ymins[i], ymins[i]], dtype=float))
        # Curved flow segment from axis i to axis i+1
        if i + 1 < n:
            xt, yt = flow_curve_boundary(
                float(xmaxs[i]), float(xmins[i + 1]),
                float(ymaxs[i]), float(ymaxs[i + 1]),
                knot_pos=knot_pos, knot_prop=knot_prop,
                curve_type=curve_type, curve_range=curve_range, segments=segments,
            )
            xb, yb = flow_curve_boundary(
                float(xmaxs[i]), float(xmins[i + 1]),
                float(ymins[i]), float(ymins[i + 1]),
                knot_pos=knot_pos, knot_prop=knot_prop,
                curve_type=curve_type, curve_range=curve_range, segments=segments,
            )
            # Skip duplicate endpoints
            top_x.append(xt[1:-1])
            top_y.append(yt[1:-1])
            bot_x.append(xb[1:-1])
            bot_y.append(yb[1:-1])

    tx = np.concatenate(top_x)
    ty = np.concatenate(top_y)
    bx = np.concatenate(bot_x)
    by = np.concatenate(bot_y)
    xs = np.concatenate([tx, bx[::-1]])
    ys = np.concatenate([ty, by[::-1]])
    return xs, ys
