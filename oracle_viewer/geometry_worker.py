"""Isolated FreeCAD worker; STEP exact shapes and per-face display meshes.
Run with system Python containing FreeCAD, never inside the model runtime.
"""
from __future__ import annotations
import hashlib
import json
import math
import sys
from pathlib import Path


def convert(source: Path, target: Path, deflection_mm: float = 0.08) -> dict:
    sys.path.append('/usr/lib/freecad-python3/lib')
    import FreeCAD
    import Part
    shape = Part.read(str(source))
    if shape.isNull() or not shape.Faces:
        raise ValueError('CAD contains no faces')
    if not shape.isValid():
        raise ValueError('CAD B-Rep validation failed; repair source before binding FAI')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    solids = shape.Solids or [shape]
    faces, triangles = [], 0
    for body_no, body in enumerate(solids, 1):
        for face_no, face in enumerate(body.Faces, 1):
            vertices, indices = face.tessellate(deflection_mm)
            triangles += len(indices)
            if triangles > 1500000:
                raise ValueError('Display mesh exceeds 1.5M triangles')
            positions = [round(v, 7) for p in vertices for v in (p.x, p.y, p.z)]
            if not all(math.isfinite(v) for v in positions):
                raise ValueError('Non-finite geometry')
            center = face.CenterOfMass
            record = {'id': f'{digest[:16]}:body{body_no}:face{face_no}',
                      'body': body_no, 'face': face_no, 'surface': type(face.Surface).__name__,
                      'center': [center.x, center.y, center.z], 'area_mm2': face.Area,
                      'positions': positions, 'indices': [v for tri in indices for v in tri]}
            for name in ('Radius', 'Axis'):
                if hasattr(face.Surface, name):
                    value = getattr(face.Surface, name)
                    record[name.lower()] = ([value.x, value.y, value.z]
                                             if hasattr(value, 'x') else float(value))
            faces.append(record)
    bbox = shape.BoundBox
    output = {'schema': 'oracle-face-mesh-v1', 'source_sha256': digest,
              'source_format': source.suffix.lstrip('.').upper(), 'units': 'mm', 'brep_valid': True,
              'geometry_authority': 'source CAD; display mesh is not metrology geometry',
              'face_ids_scope': 'source SHA256; reimport/revision requires revalidation',
              'meshing_deflection_mm': deflection_mm, 'body_count': len(solids),
              'face_count': len(faces), 'triangle_count': triangles,
              'bbox': [bbox.XMin, bbox.YMin, bbox.ZMin, bbox.XMax, bbox.YMax, bbox.ZMax],
              'volume_mm3': shape.Volume, 'faces': faces}
    target.write_text(json.dumps(output, separators=(',', ':'), allow_nan=False))
    return {k: v for k, v in output.items() if k != 'faces'}


if __name__ == '__main__':
    try:
        result = convert(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.08)
        print('ORACLE_RESULT=' + json.dumps(result, allow_nan=False))
    except Exception as exc:
        print('ORACLE_ERROR=' + str(exc))
        sys.exit(1)
