"""Local X_T/X_B decoding and display meshes, isolated from the LLM runtime.

This is a display importer. CHART interpolation and pipe construction do not
claim an exact, metrologically qualified Parasolid-to-OCCT conversion.
"""

from __future__ import annotations
import hashlib
import json
import math
import sys
from pathlib import Path

DISPLAY_LINEAR_DEFLECTION_MM = 0.01
DISPLAY_ANGULAR_DEFLECTION_RAD = 0.12


def remove_degenerate_triangles(record):
    """Drop only triangles whose 3D area is at floating-point noise scale."""
    positions = record["positions"]
    indices = record["indices"]
    if len(positions) % 3 or len(indices) % 3:
        raise ValueError("Display mesh has malformed vertex or triangle arrays")
    count = len(positions) // 3
    if any(index < 0 or index >= count for index in indices):
        raise ValueError("Display mesh contains an out-of-range vertex index")
    if not indices:
        raise ValueError("Display mesh contains no triangles")
    extents = [
        max(positions[axis::3]) - min(positions[axis::3])
        for axis in range(3)
    ]
    diagonal = math.sqrt(sum(value * value for value in extents))
    cross_limit = max(diagonal**2 * 1e-12, 1e-15)
    cross_limit_sq = cross_limit * cross_limit
    kept = []
    removed = 0
    for offset in range(0, len(indices), 3):
        a, b, c = indices[offset : offset + 3]
        if a == b or b == c or c == a:
            removed += 1
            continue
        ax, ay, az = positions[a * 3 : a * 3 + 3]
        ux, uy, uz = (positions[b * 3 + i] - v for i, v in enumerate((ax, ay, az)))
        vx, vy, vz = (positions[c * 3 + i] - v for i, v in enumerate((ax, ay, az)))
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        if nx * nx + ny * ny + nz * nz <= cross_limit_sq:
            removed += 1
            continue
        kept.extend((a, b, c))
    record["indices"] = kept
    record["degenerate_triangles_removed"] = removed
    return record


def decoded_fields(node):
    return {
        f["definition"]["name"]: [v["value"] for v in f["values"]]
        for f in node["fields"]
    }


def instantiate(prototypes, brep, raw_nodes, digest):
    """Apply XT assembly transforms once; preserve repeated part instances."""
    bodies = {b.source.node_index: b.id for b in brep.bodies}
    body_for_face = {}
    for body in brep.bodies:
        for rid in body.regions:
            for sid in brep.regions[rid].shells:
                shell = brep.shells[sid]
                for fid in (*shell.back_faces, *shell.front_faces):
                    if fid in body_for_face and body_for_face[fid] != body.id:
                        raise ValueError("Face belongs to multiple source bodies")
                    body_for_face[fid] = body.id
    by_body = {b.id: [] for b in brep.bodies}
    for face in prototypes:
        by_body[body_for_face[face["source_face_id"]]].append(face)
    nodes = {n["index"]: n for n in raw_nodes}
    assemblies = [n for n in raw_nodes if n["node_type"] == 10]
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    instances = []

    def multiply(parent, child):
        # Packed affine rows: nine rotation/scale values, then mm translation.
        a, b = parent[:9], child[:9]
        return tuple(
            sum(a[r * 3 + k] * b[k * 3 + c] for k in range(3))
            for r in range(3)
            for c in range(3)
        ) + tuple(
            sum(a[r * 3 + k] * child[9 + k] for k in range(3)) + parent[9 + r]
            for r in range(3)
        )

    def placement(index):
        if not index:
            return identity
        f = decoded_fields(nodes[index])
        if nodes[index]["node_type"] != 100:
            raise ValueError("Instance references a non-transform node")
        scale = f["scale"][0]
        values = tuple(x * scale for x in f["rotation_matrix"]) + tuple(
            x * 1000 * scale for x in f["translation_vector"][0]
        )
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Invalid assembly transform")
        return values

    referenced = {
        decoded_fields(n)["part"][0] for n in raw_nodes if n["node_type"] == 11
    }
    roots = [n["index"] for n in assemblies if n["index"] not in referenced]

    def visit(part, transform, stack, path):
        if part in stack:
            raise ValueError("Cyclic assembly reference")
        if part in bodies:
            instances.append((bodies[part], transform, path))
            return
        node = nodes.get(part)
        if node is None or node["node_type"] != 10:
            raise ValueError("Unsupported assembly part reference")
        pointer = decoded_fields(node)["sub_instance"][0]
        seen = set()
        while pointer:
            if pointer in seen:
                raise ValueError("Cyclic instance list")
            seen.add(pointer)
            child = nodes[pointer]
            if child["node_type"] != 11:
                raise ValueError("Invalid assembly instance")
            f = decoded_fields(child)
            visit(
                f["part"][0],
                multiply(transform, placement(f["transform"][0])),
                stack | {part},
                path + (pointer,),
            )
            pointer = f["next_in_part"][0]

    if assemblies:
        if not roots:
            raise ValueError("Assembly has no root")
        for root in roots:
            visit(root, identity, set(), (root,))
    else:
        instances = [(b.id, identity, (b.source.node_index,)) for b in brep.bodies]
    if {body for body, _, _ in instances} != set(by_body):
        raise ValueError("Unplaced source bodies; assembly coverage is incomplete")
    faces = []
    for number, (body, m, path) in enumerate(instances, 1):
        for prototype in by_body[body]:
            fid = prototype["source_face_id"]
            positions = []
            for i in range(0, len(prototype["positions"]), 3):
                p = prototype["positions"][i : i + 3]
                positions.extend(
                    round(sum(m[r * 3 + k] * p[k] for k in range(3)) + m[9 + r], 7)
                    for r in range(3)
                )
            count = len(positions) // 3
            center = [sum(positions[k::3]) / count for k in range(3)]
            # A mirrored assembly placement reverses triangle winding.
            determinant = (
                m[0] * (m[4] * m[8] - m[5] * m[7])
                - m[1] * (m[3] * m[8] - m[5] * m[6])
                + m[2] * (m[3] * m[7] - m[4] * m[6])
            )
            indices = prototype["indices"]
            if determinant < 0:
                indices = [
                    v
                    for i in range(0, len(indices), 3)
                    for v in (indices[i], indices[i + 2], indices[i + 1])
                ]
            faces.append(
                {
                    "id": f"{digest[:16]}:instance{number}:face{fid}",
                    "body": number,
                    "prototype_body": body,
                    "face": fid,
                    "source_node_index": brep.faces[fid].source.node_index,
                    "instance_path": list(path),
                    "surface": brep.surfaces[brep.faces[fid].surface].kind.value,
                    "center": center,
                    "positions": positions,
                    "indices": indices,
                    "degenerate_triangles_removed": prototype.get(
                        "degenerate_triangles_removed", 0
                    ),
                    "display_construction": prototype.get(
                        "display_construction", "OCCT tessellation"
                    ),
                    "occt_face_valid": prototype.get("occt_face_valid"),
                }
            )
    return faces, len(instances)


def triangulation_record(face, fid):
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location

    location = TopLoc_Location()
    mesh = BRep_Tool.Triangulation_s(face, location)
    if mesh is None or not mesh.NbTriangles():
        return None
    positions = [
        v
        for i in range(1, mesh.NbNodes() + 1)
        for v in mesh.Node(i).Transformed(location.Transformation()).Coord()
    ]
    indices = []
    for i in range(1, mesh.NbTriangles() + 1):
        a, b, c = mesh.Triangle(i).Get()
        indices.extend(
            (a - 1, c - 1, b - 1)
            if face.Orientation() == TopAbs_REVERSED
            else (a - 1, b - 1, c - 1)
        )
    return {"source_face_id": fid, "positions": positions, "indices": indices}


def periodic_band(builder, source_face, shape):
    """Recreate a full circular band from its two source circle boundaries."""
    from parasolid_kit.brep import CircleCurve, ConeSurface, CylinderSurface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_REVERSED

    surface = builder.surfaces[source_face.surface].definition
    if (
        not isinstance(surface, (ConeSurface, CylinderSurface))
        or len(source_face.loops) != 2
    ):
        return None
    bounds = []
    for lid in source_face.loops:
        half_edges = builder.loops[lid].half_edges
        if len(half_edges) != 1:
            return None
        edge = builder.edges[builder.half_edges[half_edges[0]].edge]
        if (
            edge.start_vertex is not None
            or edge.end_vertex is not None
            or edge.curve is None
        ):
            return None
        curve = builder.curves[edge.curve].definition
        if not isinstance(curve, CircleCurve):
            return None
        v = (
            sum(
                (x - y) * a
                for x, y, a in zip(curve.center, surface.point, surface.axis)
            )
            * builder.geometry.scale
        )
        bounds.append(
            v / surface.cos_half_angle if isinstance(surface, ConeSurface) else v
        )
    result = BRepBuilderAPI_MakeFace(
        builder._surface_geometry(source_face.surface),
        0,
        math.tau,
        min(bounds),
        max(bounds),
        1e-7,
    ).Face()
    if shape.Orientation() == TopAbs_REVERSED:
        result.Reverse()
    BRepMesh_IncrementalMesh(
        result,
        DISPLAY_LINEAR_DEFLECTION_MM,
        False,
        DISPLAY_ANGULAR_DEFLECTION_RAD,
        False,
    )
    record = triangulation_record(result, source_face.id)
    if record:
        record["display_construction"] = "full periodic band bounded by source circles"
    return record


def ordered_uv_rings(uv,ends):
    import numpy as np
    rings=np.split(np.asarray(uv,dtype=float),ends[:-1])
    areas=[abs(float(np.sum(r[:,0]*np.roll(r[:,1],-1)-r[:,1]*np.roll(r[:,0],-1)))) for r in rings]
    outer=max(range(len(rings)),key=lambda i:areas[i])
    if outer:
        rings=[rings[outer]]+[r for i,r in enumerate(rings) if i!=outer]
        uv=np.concatenate(rings).tolist();ends=np.cumsum([len(r) for r in rings]).tolist()
    return uv,ends


def boundary_mesh(builder, source_face, shape):
    """Triangulate source trim boundaries on the decoded parametric surface."""
    import numpy as np
    import mapbox_earcut
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.TopAbs import TopAbs_REVERSED

    surface = builder._surface_geometry(source_face.surface)
    uv = []
    ends = []
    for lid in source_face.loops:
        ring = []
        wire = BRepTools_WireExplorer(builder._make_wire(lid))
        while wire.More():
            edge = wire.Current()
            curve = BRepAdaptor_Curve(edge)
            sampler = GCPnts_QuasiUniformDeflection(curve, 0.005)
            if not sampler.IsDone():
                raise ValueError("Boundary sampling failed")
            points = [sampler.Value(j) for j in range(1, sampler.NbPoints() + 1)]
            if edge.Orientation() == TopAbs_REVERSED:
                points.reverse()
            for point in points:
                if not ring or point.Distance(ring[-1]) > 1e-7:
                    ring.append(point)
            wire.Next()
        if len(ring) < 3:
            raise ValueError("Trim ring has fewer than three points")
        if ring[-1].Distance(ring[0]) < 1e-7:
            ring.pop()
        projected = []
        for point in ring:
            operation = GeomAPI_ProjectPointOnSurf(point, surface)
            if not operation.NbPoints():
                raise ValueError("Boundary projection failed")
            u, v = operation.LowerDistanceParameters()
            if projected:
                if surface.IsUPeriodic():
                    period = surface.UPeriod()
                    u = (
                        projected[-1][0]
                        + (u - projected[-1][0] + period / 2) % period
                        - period / 2
                    )
                if surface.IsVPeriodic():
                    period = surface.VPeriod()
                    v = (
                        projected[-1][1]
                        + (v - projected[-1][1] + period / 2) % period
                        - period / 2
                    )
            projected.append((u, v))
        uv.extend(projected)
        ends.append(len(uv))
    # Parasolid LOOP lists do not promise that the outer boundary comes first.
    # Earcut does require that ordering; preserve every loop while choosing the
    # largest UV area as the outer boundary.
    uv,ends=ordered_uv_rings(uv,ends)
    indices = mapbox_earcut.triangulate_float64(
        np.asarray(uv, dtype=np.float64), np.asarray(ends, dtype=np.uint32)
    ).tolist()
    if not indices:
        raise ValueError(f"Boundary triangulation is empty: source face={source_face.id}, surface={source_face.surface}, UV extents={np.ptp(np.asarray(uv),axis=0).tolist()}")
    xyz = [surface.Value(*point).Coord() for point in uv]
    triangles = [indices[k : k + 3] for k in range(0, len(indices), 3)]
    for iteration in range(8):
        refined = []
        cache = {}
        split_count = 0

        def midpoint(a, b):
            key = tuple(sorted((a, b)))
            if key not in cache:
                point = ((np.asarray(uv[a]) + np.asarray(uv[b])) / 2).tolist()
                cache[key] = len(uv)
                uv.append(point)
                xyz.append(surface.Value(*point).Coord())
            return cache[key]

        for a, b, c in triangles:
            worst = 0
            for left, right in ((a, b), (b, c), (c, a)):
                point = surface.Value(
                    *((np.asarray(uv[left]) + np.asarray(uv[right])) / 2)
                )
                worst = max(
                    worst,
                    np.linalg.norm(
                        np.asarray(point.Coord())
                        - (np.asarray(xyz[left]) + np.asarray(xyz[right])) / 2
                    ),
                )
            if worst > 0.02:
                ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
                refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
                split_count += 1
            else:
                refined.append((a, b, c))
        triangles = refined
        if len(triangles) > 200000:
            raise ValueError("Surface refinement exceeds display mesh limit")
        if not split_count:
            break
    else:
        raise ValueError("Surface refinement did not converge")
    if shape.Orientation() == TopAbs_REVERSED:
        triangles = [(a, c, b) for a, b, c in triangles]
    return {
        "source_face_id": source_face.id,
        "positions": [v for point in xyz for v in point],
        "indices": [v for triangle in triangles for v in triangle],
        "display_construction": "source boundary triangulation on decoded surface",
    }


def repair_wire_orientation(shape):
    """Fix trim orientation only, retaining every original edge and wire.

    No new curves, tolerance inflation, missing seams or natural bounds are
    permitted. Kernel validity still does not establish metrology approval.
    """
    from OCP.ShapeFix import ShapeFix_Face
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_WIRE
    from OCP.BRep import BRep_Tool
    def entities(face,kind):
        explorer=TopExp_Explorer(face,kind);result=[]
        while explorer.More():
            result.append(explorer.Current());explorer.Next()
        return result
    original=entities(shape,TopAbs_EDGE)
    fixer=ShapeFix_Face(shape)
    fixer.SetPrecision(1e-5)
    if not fixer.FixOrientation():return None
    fixed=fixer.Face();edges=entities(fixed,TopAbs_EDGE)
    if (len(edges)!=len(original) or len(entities(shape,TopAbs_WIRE))!=len(entities(fixed,TopAbs_WIRE))
            or BRep_Tool.Surface_s(shape)!=BRep_Tool.Surface_s(fixed)):
        return None
    remaining=list(edges)
    for edge in original:
        match=next((i for i,other in enumerate(remaining) if edge.IsSame(other)),None)
        if match is None:return None
        remaining.pop(match)
    return fixed if BRepCheck_Analyzer(fixed).IsValid() else None


def face_validation_issues(shape, analyzer):
    """Preserve kernel failure categories, including edge-in-face checks."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_WIRE, TopAbs_EDGE
    from OCP.BRepCheck import BRepCheck_NoError
    result=[]
    def record(entity,kind,index):
        check=analyzer.Result(entity)
        if check is None:return
        statuses=list(check.Status())
        if kind!='face':
            from OCP.Standard import Standard_NoSuchObject
            try:statuses.extend(check.StatusOnShape(shape))
            except Standard_NoSuchObject:
                # Not every subshape has a face-context check. Its intrinsic
                # status still applies; missing context is not a kernel error.
                pass
        errors=sorted({str(s).split('.')[-1] for s in statuses if s!=BRepCheck_NoError})
        if errors:result.append(dict(kind=kind,index=index,errors=errors))
    record(shape,'face',0)
    for kind,label in ((TopAbs_WIRE,'wire'),(TopAbs_EDGE,'edge')):
        explorer=TopExp_Explorer(shape,kind);index=0
        while explorer.More():
            record(explorer.Current(),label,index);index+=1;explorer.Next()
    return result


def convert(source: Path, target: Path, *, fix_wire_orientation=False):
    from parasolid_kit import read_brep
    from parasolid_kit.interop.occt.options import (
        OcctConversionOptions,
        ValidationTolerances,
    )
    from OCP.BRep import BRep_Builder
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopoDS import TopoDS_Compound, TopoDS
    from OCP.BRepCheck import BRepCheck_Analyzer
    from parasolid_display import DisplayBuilder, DisplayGeometry

    schema_dir = (
        Path(__file__).resolve().parents[2] / "third_party" / "parasolid-schemas"
    )
    # Prefer an exact-key built-in profile when the installed parser has one.
    # If that reviewed subset does not cover this file, try the exact external
    # catalog next. Never substitute a nearby schema version.
    try:
        decoded = read_brep(source)
    except Exception as profile_error:
        try:
            from parasolid_legacy import legacy_provider
            provider = legacy_provider(source, schema_dir)
            decoded = read_brep(source, schema_provider=provider) if provider else read_brep(source, schema_dir=str(schema_dir))
        except Exception as catalog_error:
            profile_diagnostic = getattr(profile_error, "diagnostic", None)
            catalog_diagnostic = getattr(catalog_error, "diagnostic", None)
            profile_reason = (
                f"{profile_diagnostic.code}: {profile_diagnostic.message}; "
                f"details={profile_diagnostic.details}"
                if profile_diagnostic is not None else str(profile_error)
            )
            catalog_reason = (
                f"{catalog_diagnostic.code}: {catalog_diagnostic.message}; "
                f"details={catalog_diagnostic.details}"
                if catalog_diagnostic is not None else str(catalog_error)
            )
            raise ValueError(
                "内置精确版本 profile 与外部精确 schema 均未能解码此文件。"
                f" 内置 profile: {profile_reason}；外部 schema: {catalog_reason}"
            ) from catalog_error
    brep = decoded.brep
    if not brep.complete or not brep.topology.valid:
        raise ValueError("Parasolid geometry decoding or source topology is incomplete")
    source_summary = decoded.summary.to_dict()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    builder = DisplayBuilder(
        brep,
        DisplayGeometry(
            OcctConversionOptions(
                source_unit="m",
                target_unit="mm",
                validation=ValidationTolerances(linear_absolute=1e-5),
            )
        ),
    )
    builder.precision = 0.005
    for edge in brep.edges:
        if edge.curve is not None:
            for vid in (edge.start_vertex, edge.end_vertex):
                if vid is not None:
                    builder.curve_anchors[edge.curve].append(
                        builder.geometry.point(
                            builder.points[builder.vertices[vid].point].position
                        )
                    )
    builder._build_vertices()
    for edge in brep.edges:
        builder._build_edge(edge)
    for face in brep.faces:
        builder._build_face(face)
    for face in brep.faces:
        builder._finish_face(face)
    orientation_repairs=[]
    if fix_wire_orientation:
        for face in brep.faces:
            shape=TopoDS.Face_s(builder.face_shapes[face.id])
            if BRepCheck_Analyzer(shape).IsValid():continue
            fixed=repair_wire_orientation(shape)
            if fixed is not None:
                builder.face_shapes[face.id]=fixed
                orientation_repairs.append(dict(source_face_id=face.id,
                    construction='trim wire orientation only; original edges and support preserved',
                    tolerance_mm=1e-5,metrology_accepted=False))
    compound = TopoDS_Compound()
    kernel = BRep_Builder()
    kernel.MakeCompound(compound)
    for shape in builder.face_shapes.values():
        kernel.Add(compound, shape)
    # Keep the source geometry and every face. Coarsen only display
    # tessellation, with explicit bounded settings, before allocating a large
    # browser payload. Clean old triangulations so OCCT can actually remesh.
    from OCP.BRepTools import BRepTools
    face_validity = {}
    invalid_ids = []
    invalid_diagnostics = []
    for face in brep.faces:
        shape = TopoDS.Face_s(builder.face_shapes[face.id])
        analyzer = BRepCheck_Analyzer(shape)
        face_validity[face.id] = analyzer.IsValid()
        if not face_validity[face.id]:
            invalid_ids.append(face.id)
            invalid_diagnostics.append(dict(source_face_id=face.id,
                surface_kind=brep.surfaces[face.surface].kind.value,
                issues=face_validation_issues(shape, analyzer)))
    invalid = len(invalid_ids)
    raw_nodes = [
        n.to_dict() for n in decoded.document.nodes if n.node_type in (10, 11, 100)
    ]
    mesh_attempts = []
    settings = ((DISPLAY_LINEAR_DEFLECTION_MM, DISPLAY_ANGULAR_DEFLECTION_RAD),
                (DISPLAY_LINEAR_DEFLECTION_MM, 0.24),
                (DISPLAY_LINEAR_DEFLECTION_MM, 0.35),
                (DISPLAY_LINEAR_DEFLECTION_MM, 0.5))
    for linear_deflection, angular_deflection in settings:
        if mesh_attempts:
            BRepTools.Clean_s(compound)
        BRepMesh_IncrementalMesh(compound, linear_deflection, False,
                                angular_deflection, True)
        prototypes = []
        repairs = []
        for face in brep.faces:
            shape = TopoDS.Face_s(builder.face_shapes[face.id])
            record = triangulation_record(shape, face.id)
            if record is None:
                record = periodic_band(builder, face, shape) or boundary_mesh(builder, face, shape)
                repairs.append({"source_face_id": face.id,
                                "construction": record["display_construction"]})
            record['occt_face_valid'] = face_validity[face.id]
            prototypes.append(remove_degenerate_triangles(record))
        faces, count = instantiate(prototypes, brep, raw_nodes, digest)
        triangles = sum(len(face["indices"]) // 3 for face in faces)
        mesh_attempts.append(dict(linear_deflection_mm=linear_deflection,
                                  angular_deflection_rad=angular_deflection,
                                  triangle_count=triangles, source_face_count=len(prototypes)))
        print('ORACLE_PROGRESS=' + json.dumps(mesh_attempts[-1]), flush=True)
        if triangles <= 1500000:
            break
        del faces, prototypes
    else:
        raise ValueError(f"Display mesh exceeds 1.5M triangles after bounded remeshing: {triangles}")
    if not faces or not all(math.isfinite(v) for f in faces for v in f["positions"]):
        raise ValueError("Empty or non-finite display geometry")
    bbox = [min(min(f["positions"][k::3]) for f in faces) for k in range(3)] + [
        max(max(f["positions"][k::3]) for f in faces) for k in range(3)
    ]
    output = {
        "schema": "oracle-face-mesh-v1",
        "status": "preview",
        "source_sha256": digest,
        "source_format": source.suffix[1:].upper(),
        "source_units": "m",
        "units": "mm",
        "source_topology_valid": True,
        "source_decoding_complete": True,
        "source_schema_key": source_summary["schema_key"]["raw"],
        "source_schema_resolution": source_summary["schema_resolution"],
        "source_node_count": source_summary["node_count"],
        "source_edge_count": len(brep.edges),
        "source_vertex_count": len(brep.vertices),
        "source_diagnostic_count": source_summary["diagnostic_count"],
        "brep_valid": False,
        "metrology_ready": False,
        "prototype_body_count": len(brep.bodies),
        "source_face_count": len(brep.faces),
        "meshed_source_face_count": len(prototypes),
        "body_count": count,
        "face_count": len(faces),
        "triangle_count": triangles,
        "degenerate_triangle_count_removed": sum(
            face["degenerate_triangles_removed"] for face in faces
        ),
        "bbox": bbox,
        "meshing_deflection_mm": linear_deflection,
        "meshing_angular_deflection_rad": angular_deflection,
        "meshing_attempts": mesh_attempts,
        "display_construction_precision_mm": 0.005,
        "display_blend_sections": getattr(builder, "display_blend_sections", []),
        "display_circle_endpoint_tolerances": getattr(builder, "display_circle_endpoint_tolerances", []),
        "display_short_line_chord_count": getattr(
            builder, "display_short_line_chord_count", 0
        ),
        "geometry_authority": "source Parasolid geometry; CHART/pipe approximations are for display only",
        "face_ids_scope": "source SHA256 + assembly instance + source face; revision invalidates associations",
        "display_repairs": repairs,
        "wire_orientation_repairs": orientation_repairs,
        "invalid_occt_face_count": invalid,
        "invalid_occt_source_face_ids": invalid_ids,
        "invalid_occt_face_diagnostics": invalid_diagnostics,
        "reason": f"真实 {source.suffix[1:].upper()} 几何预览；所有源面已生成显示网格。严格 B-Rep/计量转换尚未通过，FAI 关联未启用。",
        "faces": faces,
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False))
    temporary.replace(target)
    return {k: v for k, v in output.items() if k != "faces"}


if __name__ == "__main__":
    try:
        print(
            "ORACLE_RESULT="
            + json.dumps(convert(Path(sys.argv[1]), Path(sys.argv[2]),
                                 fix_wire_orientation='--fix-wire-orientation' in sys.argv[3:]), allow_nan=False)
        )
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print("ORACLE_ERROR=" + str(exc))
        sys.exit(1)
