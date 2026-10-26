"""CAD-kernel regressions; run with the isolated .venv-cad interpreter."""

from math import cos, pi, sin, hypot

import pytest

pytest.importorskip("parasolid_kit")
pytest.importorskip("OCP")

from OCP.Geom import Geom_BSplineCurve, Geom_ConicalSurface, Geom_CylindricalSurface
from OCP.Geom import Geom_SurfaceOfLinearExtrusion
from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
from parasolid_kit.brep import OffsetSurface
from parasolid_kit.interop.occt.geometry import GeometryFactory
from parasolid_kit.interop.occt.intersection import fit_intersection, project_surface
from parasolid_kit.interop.occt.options import OcctConversionOptions


def profile(crease=False):
    poles = TColgp_Array1OfPnt(1, 7)
    for i, x in enumerate((0, 1, 2, 3, 5, 6, 7), 1):
        poles.SetValue(i, gp_Pnt(x, (x - 3) if crease and i > 4 else 0, 0))
    knots, mults = TColStd_Array1OfReal(1, 3), TColStd_Array1OfInteger(1, 3)
    for i, (knot, mult) in enumerate(((0, 4), (0.5, 3), (1, 4)), 1):
        knots.SetValue(i, knot)
        mults.SetValue(i, mult)
    return Geom_SurfaceOfLinearExtrusion(
        Geom_BSplineCurve(poles, knots, mults, 3), gp_Dir(0, 0, 1)
    )


def test_offset_accepts_g1_join_with_different_derivative_speeds():
    basis = profile()
    factory = GeometryFactory(OcctConversionOptions(source_unit="mm"))
    surface = factory.surface3d(OffsetSurface(1, 0.2), resolve_basis=lambda _: basis)
    for u in (0.25, 0.5, 0.75):
        p = surface.Value(u, 1)
        assert p.Y() == pytest.approx(-0.2, abs=1e-12)
        assert p.Z() == pytest.approx(1)
        assert p.X() == pytest.approx(basis.Value(u, 1).X())


def test_wire_orientation_fix_preserves_hole_and_original_boundary_geometry():
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from oracle_viewer.parasolid_worker import repair_wire_orientation, face_validation_issues
    def wire(points):
        polygon=BRepBuilderAPI_MakePolygon()
        for x,y in points:polygon.Add(gp_Pnt(x,y,0))
        polygon.Close();return polygon.Wire()
    maker=BRepBuilderAPI_MakeFace(wire([(0,0),(10,0),(10,10),(0,10)]))
    maker.Add(wire([(2,2),(4,2),(4,4),(2,4)]))
    original=maker.Face()
    assert not BRepCheck_Analyzer(original).IsValid()
    issues=face_validation_issues(original,BRepCheck_Analyzer(original))
    assert any('BRepCheck_BadOrientationOfSubshape' in row['errors'] for row in issues)
    fixed=repair_wire_orientation(original)
    assert fixed is not None and BRepCheck_Analyzer(fixed).IsValid()
    props=GProp_GProps();BRepGProp.SurfaceProperties_s(fixed,props)
    assert props.Mass()==pytest.approx(96)
    assert face_validation_issues(fixed,BRepCheck_Analyzer(fixed))==[]


def test_offset_rejects_actual_profile_crease():
    factory = GeometryFactory(OcctConversionOptions(source_unit="mm"))
    with pytest.raises(ValueError, match="non-G1 join"):
        factory.surface3d(OffsetSurface(1, 0.2), resolve_basis=lambda _: profile(True))


def test_nearly_flat_cone_closed_intersection_preserves_source_surfaces():
    # OCCT's apex-based projection is ill conditioned for this source angle.
    angle = pi / 2 - 4.549575e-8
    frame = gp_Ax3(gp_Pnt(30, 55, 2), gp_Dir(0, 0, 1))
    cone = Geom_ConicalSurface(frame, angle, 0.2)
    cylinder = Geom_CylindricalSurface(frame, 2)
    height = 1.8 * cos(angle) / sin(angle)
    samples = [
        gp_Pnt(30 + 2 * cos(2 * pi * i / 19), 55 + 2 * sin(2 * pi * i / 19), 2 + height)
        for i in range(20)
    ]
    tolerance = 1e-5
    curve = fit_intersection([cylinder, cone], samples, [], tolerance)
    assert curve.IsPeriodic()
    for i in range(1001):
        p = curve.Value(curve.FirstParameter() + curve.Period() * i / 1000)
        radius = hypot(p.X() - 30, p.Y() - 55)
        assert abs(radius - 2) <= tolerance
        # Independent signed distance in the cone's meridian, without apex.
        assert abs((radius - 0.2) * cos(angle) - (p.Z() - 2) * sin(angle)) <= tolerance


def test_cone_projection_handles_negative_nappe_without_replacing_angle():
    angle = 0.7
    cone = Geom_ConicalSurface(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), angle, 1)
    p = cone.Value(0.4, -4)
    q, normal, distance = project_surface(p, cone, 1e-12)
    assert distance < 1e-12
    assert q.Distance(p) < 1e-12
    assert sum(x * x for x in normal) == pytest.approx(1)


def test_display_triangle_cleanup_drops_only_collapsed_facets():
    from oracle_viewer.parasolid_worker import remove_degenerate_triangles

    record = {
        "positions": [0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0],
        "indices": [0, 1, 2, 0, 1, 3],
    }
    result = remove_degenerate_triangles(record)
    assert result["indices"] == [0, 1, 2]
    assert result["degenerate_triangles_removed"] == 1
    collapsed = remove_degenerate_triangles({
        "positions": [0, 0, 0, 1, 0, 0, 2, 0, 0],
        "indices": [0, 1, 2],
    })
    assert collapsed["indices"] == []
    assert collapsed["degenerate_triangles_removed"] == 1


def test_line_endpoint_check_respects_explicit_source_tolerance():
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from parasolid_kit.brep import LineCurve, Vector3

    factory = GeometryFactory(OcctConversionOptions(source_unit="mm"))
    line = LineCurve(Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0))

    def build(error, tolerance):
        start, end = Vector3(0.0, error, 0.0), Vector3(1.0, error, 0.0)
        return factory.make_line_edge(
            line,
            BRepBuilderAPI_MakeVertex(factory.point(start)).Vertex(),
            BRepBuilderAPI_MakeVertex(factory.point(end)).Vertex(),
            start,
            end,
            source_tolerance=tolerance,
        )

    with pytest.raises(ValueError, match="off its curve"):
        build(4e-6, 0)
    assert not build(4e-6, 1e-5).IsNull()
    with pytest.raises(ValueError, match="off its curve"):
        build(2e-5, 1e-5)


def tensor_profile(crease=False):
    from OCP.Geom import Geom_BSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt

    poles = TColgp_Array2OfPnt(1, 2, 1, 7)
    for u in (1, 2):
        for v, x in enumerate((0, 1, 2, 3, 5, 6, 7), 1):
            poles.SetValue(u, v, gp_Pnt(x, (x - 3) if crease and v > 4 else 0, u - 1))
    uk, um = TColStd_Array1OfReal(1, 2), TColStd_Array1OfInteger(1, 2)
    vk, vm = TColStd_Array1OfReal(1, 3), TColStd_Array1OfInteger(1, 3)
    for i, value in enumerate((0.0, 1.0), 1):
        uk.SetValue(i, value)
        um.SetValue(i, 2)
    for i, (value, mult) in enumerate(((0.0, 4), (0.5, 3), (1.0, 4)), 1):
        vk.SetValue(i, value)
        vm.SetValue(i, mult)
    return Geom_BSplineSurface(poles, uk, vk, um, vm, 1, 3)


def test_offset_tensor_certifies_proportional_derivative_control_net():
    basis = tensor_profile()
    factory = GeometryFactory(OcctConversionOptions(source_unit="mm"))
    surface = factory.surface3d(OffsetSurface(1, 0.2), resolve_basis=lambda _: basis)
    for u in (0.1, 0.6, 0.9):
        for v in (0.25, 0.5, 0.75):
            p = surface.Value(u, v)
            assert p.Y() == pytest.approx(0.2, abs=1e-12)
            assert p.Z() == pytest.approx(u)
            assert p.X() == pytest.approx(basis.Value(u, v).X())


def test_offset_tensor_rejects_non_g1_control_net():
    factory = GeometryFactory(OcctConversionOptions(source_unit="mm"))
    with pytest.raises(ValueError, match="non-G1"):
        factory.surface3d(
            OffsetSurface(1, 0.2), resolve_basis=lambda _: tensor_profile(True)
        )


def test_profile_continuation_preserves_values_and_derivatives():
    from oracle_viewer.parasolid_domains import continue_profile
    from OCP.gp import gp_Vec

    original = profile().BasisCurve()
    extended = continue_profile(original, -0.2, 1.3)
    assert extended.FirstParameter() == pytest.approx(-0.2)
    assert extended.LastParameter() == pytest.approx(1.3)
    for i in range(301):
        u = -0.2 + 1.5 * i / 300
        assert extended.Value(u).Distance(original.Value(u)) < 1e-10
        a, b, da, db = gp_Pnt(), gp_Pnt(), gp_Vec(), gp_Vec()
        original.D1(u, a, da)
        extended.D1(u, b, db)
        assert (da - db).Magnitude() < 1e-9


def test_profile_continuation_retains_a_real_crease():
    from oracle_viewer.parasolid_domains import continue_profile
    from parasolid_kit.interop.occt.geometry import _check_offset_profile_g1

    extended = continue_profile(profile(True).BasisCurve(), -0.2, 1.3)
    with pytest.raises(ValueError, match="non-G1"):
        _check_offset_profile_g1(
            Geom_SurfaceOfLinearExtrusion(extended, gp_Dir(0, 0, 1))
        )
