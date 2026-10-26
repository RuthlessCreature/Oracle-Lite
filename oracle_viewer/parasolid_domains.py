"""Exact polynomial continuation of a clamped non-rational spline profile.

An explicitly supplied interval is required. This preserves the spline's
original parameterisation and polynomial end pieces; it does not select or
approve the physical parameter domain of a source surface.
"""

from __future__ import annotations

from math import comb, isfinite


def bernstein_interval(poles, start, end):
    """Bezier controls after t = start + (end-start)*s, including extrapolation."""
    degree = len(poles) - 1
    dimension = len(poles[0])
    # Power coefficients from forward differences of the Bernstein controls.
    power = [
        [
            comb(degree, k)
            * sum((-1) ** (k - j) * comb(k, j) * poles[j][axis] for j in range(k + 1))
            for axis in range(dimension)
        ]
        for k in range(degree + 1)
    ]
    width = end - start
    transformed = [
        [
            width**k
            * sum(
                comb(j, k) * start ** (j - k) * power[j][axis]
                for j in range(k, degree + 1)
            )
            for axis in range(dimension)
        ]
        for k in range(degree + 1)
    ]
    return [
        tuple(
            sum(
                comb(i, k) / comb(degree, k) * transformed[k][axis]
                for k in range(i + 1)
            )
            for axis in range(dimension)
        )
        for i in range(degree + 1)
    ]


def continue_profile(curve, first, last):
    """Extend end polynomials without changing any value in the original domain.

    Only open, clamped, non-rational B-splines are supported. Real creases in
    the original curve remain; no smoothing or tolerance increase is applied.
    """
    from OCP.Geom import Geom_BSplineCurve
    from OCP.GeomConvert import GeomConvert_BSplineCurveToBezierCurve
    from OCP.TColgp import TColgp_Array1OfPnt
    from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
    from OCP.gp import gp_Pnt

    if not isinstance(curve, Geom_BSplineCurve):
        raise ValueError("Profile continuation requires a B-spline")
    if not all(isfinite(x) for x in (first, last)) or first >= last:
        raise ValueError("Profile continuation requires a finite ordered interval")
    if curve.IsPeriodic() or curve.IsClosed() or curve.IsRational():
        raise ValueError("Profile continuation requires an open non-rational spline")
    degree = curve.Degree()
    knots = [curve.Knot(i) for i in range(1, curve.NbKnots() + 1)]
    mults = [curve.Multiplicity(i) for i in range(1, curve.NbKnots() + 1)]
    if mults[0] != degree + 1 or mults[-1] != degree + 1:
        raise ValueError("Profile continuation requires clamped end knots")
    poles = [curve.Pole(i).Coord() for i in range(1, curve.NbPoles() + 1)]
    pieces = GeomConvert_BSplineCurveToBezierCurve(curve)
    if first < knots[0]:
        piece = pieces.Arc(1)
        controls = [piece.Pole(i).Coord() for i in range(1, degree + 2)]
        extra = bernstein_interval(
            controls, (first - knots[0]) / (knots[1] - knots[0]), 0.0
        )
        poles = extra[:-1] + poles
        mults[0] = degree
        knots.insert(0, first)
        mults.insert(0, degree + 1)
    if last > knots[-1]:
        piece = pieces.Arc(pieces.NbArcs())
        controls = [piece.Pole(i).Coord() for i in range(1, degree + 2)]
        extra = bernstein_interval(
            controls, 1.0, (last - knots[-2]) / (knots[-1] - knots[-2])
        )
        poles = poles + extra[1:]
        mults[-1] = degree
        knots.append(last)
        mults.append(degree + 1)
    if not all(isfinite(x) for p in poles for x in p):
        raise ValueError("Profile continuation generated non-finite control points")
    points = TColgp_Array1OfPnt(1, len(poles))
    values = TColStd_Array1OfReal(1, len(knots))
    multiplicities = TColStd_Array1OfInteger(1, len(mults))
    for i, p in enumerate(poles, 1):
        points.SetValue(i, gp_Pnt(*p))
    for i, k in enumerate(knots, 1):
        values.SetValue(i, k)
    for i, m in enumerate(mults, 1):
        multiplicities.SetValue(i, m)
    return Geom_BSplineCurve(points, values, multiplicities, degree)
