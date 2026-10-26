"""Display construction for decoded Parasolid geometry.

Intersection CHART samples identify the source branch; interpolating these
samples is a display approximation, never a certified exact intersection.
Cliff-edge equal-radius envelopes use their decoded spine and radius. Offset
surfaces retain C0 parameter boundaries for display; strict OCCT validation is
reported separately by the worker. No source entities are silently discarded.
"""

from parasolid_kit.brep import IntersectionCurve, OffsetSurface
from parasolid_kit.interop.occt.parametric import ParametricTopologyBuilder
from parasolid_kit.interop.occt.geometry import GeometryFactory
from parasolid_kit.interop.occt.intersection import project_curve


class DisplayGeometry(GeometryFactory):
    def blend_surface(self, surface, resolve_curve):
        from dataclasses import replace
        from parasolid_kit.brep import BlendType

        return super().blend_surface(
            replace(surface, blend_type=BlendType.ROLLING_BALL), resolve_curve
        )

    def surface3d(self, definition, *, resolve_basis, resolve_curve=None):
        if isinstance(definition, OffsetSurface):
            from OCP.Geom import Geom_OffsetSurface

            return Geom_OffsetSurface(
                resolve_basis(definition.basis_surface),
                definition.offset * self.scale,
                True,
            )
        return super().surface3d(
            definition, resolve_basis=resolve_basis, resolve_curve=resolve_curve
        )


class DisplayBuilder(ParametricTopologyBuilder):
    def _constant_blend_section(self, edge):
        """Recover a constant-u rolling-ball section, bounded by source endpoints.

        The source radius and FIN UV curve establish the circle; the decoded
        spine resolves the two possible centers. Never reuse OCCT pipe UVs.
        Other blend parameter curves remain unsupported rather than guessed.
        """
        import numpy as np
        from parasolid_kit.brep import BlendedEdgeSurface, NurbsCurve, BlendType
        from OCP.gp import gp_Pnt
        from OCP.GC import GC_MakeArcOfCircle
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        if edge.start_vertex is None or edge.end_vertex is None:return False
        for hid in edge.half_edges:
            trim=self._surface_trim(self.half_edges[hid].curve)
            if not trim:continue
            support=self.surfaces[trim[1].surface].definition
            if not isinstance(support,BlendedEdgeSurface) or support.blend_type!=BlendType.ROLLING_BALL:continue
            uv=self.curves[trim[1].parameter_curve].definition
            if not isinstance(uv,NurbsCurve) or uv.degree!=1 or uv.rational or uv.vertex_dimension!=2:continue
            controls=np.asarray(uv.control_vertices,float)
            if len(controls)!=2 or np.ptp(controls[:,0])>1e-10 or abs(np.ptp(controls[:,1])-1)>1e-8:continue
            radii=np.abs(np.asarray(support.ranges,float))*self.geometry.scale
            if min(radii)<=0 or abs(radii[0]-radii[1])>1e-8 or support.thumb_weights!=(1.0,1.0):continue
            radius=float(radii[0])
            points=[self.geometry.point(self.points[self.vertices[v].point].position) for v in (edge.start_vertex,edge.end_vertex)]
            a,b=[np.asarray(p.Coord()) for p in points];chord=b-a;length=float(np.linalg.norm(chord))
            if not 1e-9<length<2*radius:continue
            spine=self._curve_geometry(support.spine_curve)
            trials=[]
            for point in points:
                t,_=project_curve(spine,point);center=np.asarray(spine.Value(t).Coord())
                tangent=np.asarray(spine.DN(t,1).Coord());norm=np.linalg.norm(tangent)
                if norm<1e-12:continue
                tangent/=norm
                direction=np.cross(tangent,chord);norm=np.linalg.norm(direction)
                if norm<1e-12:continue
                direction/=norm
                for sign in (-1,1):
                    candidate=(a+b)/2+sign*np.sqrt(radius**2-length**2/4)*direction
                    deviation=float(np.linalg.norm(candidate-center))
                    if deviation<=self.precision and abs(float(chord@tangent))<=self.precision:
                        trials.append((deviation,candidate))
            if not trials:continue
            error,center=min(trials,key=lambda row:row[0]);radial=(a-center)+(b-center)
            if np.linalg.norm(radial)<1e-10:continue
            mid=center+radius*radial/np.linalg.norm(radial)
            arc=GC_MakeArcOfCircle(points[0],gp_Pnt(*mid),points[1])
            if not arc.IsDone():continue
            op=BRepBuilderAPI_MakeEdge(arc.Value(),self.vertex_shapes[edge.start_vertex],self.vertex_shapes[edge.end_vertex])
            if not op.IsDone():continue
            self.edge_shapes[edge.id]=op.Edge()
            self.operations.append('display_constant_u_blend_section_source_radius')
            records=getattr(self,'display_blend_sections',[])
            records.append(dict(edge_id=edge.id,surface_id=trim[1].surface,radius_mm=radius,spine_center_deviation_mm=error))
            self.display_blend_sections=records
            return True
        return False

    def _curve_geometry(self, curve_id):
        if curve_id in self.curve_geometries:
            return self.curve_geometries[curve_id]
        d = self.curves[curve_id].definition
        if not isinstance(d, IntersectionCurve):
            return super()._curve_geometry(curve_id)
        from OCP.GeomAPI import GeomAPI_Interpolate
        from OCP.TColgp import TColgp_HArray1OfPnt

        points = []
        for v in (*d.start_points[:1], *d.chart_points, *d.end_points[:1]):
            q = self.geometry.point(v)
            if not points or q.Distance(points[-1]) > 1e-7:
                points.append(q)
        closed = len(points) > 2 and points[0].Distance(points[-1]) < 1e-7
        if closed:
            points.pop()
        if len(points) < 2:
            raise ValueError(f"Intersection {curve_id} has insufficient chart geometry")

        def fit(points):
            a = TColgp_HArray1OfPnt(1, len(points))
            for i, q in enumerate(points, 1):
                a.SetValue(i, q)
            op = GeomAPI_Interpolate(a, closed, 1e-9)
            op.Perform()
            if not op.IsDone():
                raise ValueError("Chart interpolation failed")
            return op.Curve()

        result = fit(points)
        if self.curve_anchors[curve_id]:
            ordered = [(project_curve(result, q)[0], q) for q in points]
            for q in self.curve_anchors[curve_id]:
                t, dist = project_curve(result, q)
                if any(q.Distance(p) < 1e-7 for _, p in ordered):
                    continue
                ordered.append((t, q))
            points = [q for _, q in sorted(ordered, key=lambda row: row[0])]
            result = fit(points)
        self.curve_geometries[curve_id] = result
        self.approximated_curves[curve_id] = self.precision
        return result

    def _build_edge(self, edge):
        from dataclasses import replace
        from parasolid_kit.brep import BlendedEdgeSurface, BlendBoundarySurface, LineCurve

        if edge.curve is None:
            candidates = []
            for hid in edge.half_edges:
                trimmed = self._surface_trim(self.half_edges[hid].curve)
                if not trimmed:
                    continue
                support = self.surfaces[trimmed[1].surface].definition
                while isinstance(support, OffsetSurface):
                    support = self.surfaces[support.basis_surface].definition
                if not isinstance(support, (BlendedEdgeSurface, BlendBoundarySurface)):
                    candidates.append(hid)
            if candidates:
                order = tuple(candidates) + tuple(
                    h for h in edge.half_edges if h not in candidates
                )
                edge = replace(edge, half_edges=order)
        try:
            return super()._build_edge(edge)
        except ValueError as exc:
            if 'BRepBuilderAPI_DifferentsPointAndParameter' in str(exc) and edge.curve is not None:
                from parasolid_kit.brep import TrimmedCurve, CircleCurve, LineCurve
                from OCP.BRep import BRep_Builder, BRep_Tool
                trim=self.curves[edge.curve].definition
                if isinstance(trim,TrimmedCurve) and isinstance(self.curves[trim.basis_curve].definition,(CircleCurve,LineCurve)):
                    curve=self._curve_geometry(trim.basis_curve)
                    factor=self.geometry.scale if isinstance(self.curves[trim.basis_curve].definition,LineCurve) else 1
                    expected=[curve.Value(t*factor) for t in (trim.start_parameter,trim.end_parameter)]
                    vertices=[self.vertex_shapes[v] for v in (edge.start_vertex,edge.end_vertex)]
                    distances=[min(q.Distance(BRep_Tool.Pnt_s(v)) for q in expected) for v in vertices]
                    if max(distances)<=self.precision:
                        for vertex,distance in zip(vertices,distances):
                            BRep_Builder().UpdateVertex(vertex,max(BRep_Tool.Tolerance_s(vertex),distance+1e-7))
                        super()._build_edge(edge)
                        records=getattr(self,'display_circle_endpoint_tolerances',[])
                        records.append(dict(edge_id=edge.id,max_deviation_mm=max(distances)))
                        self.display_circle_endpoint_tolerances=records
                        return
            if 'unsupported parameter-curve support surface' in str(exc):
                if self._constant_blend_section(edge):return
                refs=[]
                for hid in edge.half_edges:
                    trimmed=self._surface_trim(self.half_edges[hid].curve)
                    if trimmed:
                        refs.append(dict(fin=hid,surface_type=type(self.surfaces[trimmed[1].surface].definition).__name__,surface=trimmed[1].surface,original_curve=trimmed[1].original_curve))
                raise ValueError(f'{exc}; edge={edge.id}; source FIN supports={refs}') from exc
            # A short but nonzero Parasolid line may be shorter than the
            # source vertex tolerance. It is still useful for display topology;
            # retain its exact endpoint chord and record the bounded fallback.
            if not (
                isinstance(exc, ValueError)
                and "line edge endpoints must be distinct" in str(exc)
                and edge.curve is not None
                and isinstance(self.curves[edge.curve].definition, LineCurve)
                and edge.start_vertex is not None
                and edge.end_vertex is not None
            ):
                raise
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

            start = self.geometry.point(
                self.points[self.vertices[edge.start_vertex].point].position
            )
            end = self.geometry.point(
                self.points[self.vertices[edge.end_vertex].point].position
            )
            if start.Distance(end) <= 1e-12:
                raise ValueError(
                    f"line edge {edge.id} has coincident endpoints even at display precision"
                ) from exc
            fallback = BRepBuilderAPI_MakeEdge(
                self.vertex_shapes[edge.start_vertex],
                self.vertex_shapes[edge.end_vertex],
            )
            if not fallback.IsDone():
                raise ValueError(
                    f"line edge {edge.id} short display chord could not be built"
                ) from exc
            self.edge_shapes[edge.id] = fallback.Edge()
            self.operations.append("display_short_line_chord_within_source_tolerance")
            self.display_short_line_chord_count = (
                getattr(self, "display_short_line_chord_count", 0) + 1
            )
            return

    def _add_pcurve(self, half_edge, face):
        from parasolid_kit.brep import BlendedEdgeSurface

        trimmed = self._surface_trim(half_edge.curve)
        if trimmed and isinstance(
            self.surfaces[trimmed[1].surface].definition, BlendedEdgeSurface
        ):
            from OCP.ShapeFix import ShapeFix_Edge
            from OCP.BRep import BRep_Builder
            from OCP.BRepLib import BRepLib

            shape = self.edge_shapes[half_edge.edge]
            ShapeFix_Edge().FixAddPCurve(shape, face, False, self.precision)
            BRep_Builder().SameParameter(shape, False)
            BRepLib.SameParameter_s(shape, self.precision)
            return
        return super()._add_pcurve(half_edge, face)
