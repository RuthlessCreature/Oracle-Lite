"""Build a visibly-labeled outer-envelope STEP draft from drawing dimensions.

Run with the system FreeCAD Python used by oracle_viewer/geometry_worker.py.
This is not a replacement for the drawing's referenced master geometric data.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.append('/usr/lib/freecad-python3/lib')
import FreeCAD as App
import Part


def rounded_rectangle(width: float, height: float, radius: float, z: float = 0.0):
    hx, hy = width / 2.0, height / 2.0
    r = radius
    sqrt2 = math.sqrt(2.0)
    v = lambda x, y: App.Vector(x, y, z)
    edges = [
        Part.makeLine(v(-hx + r, -hy), v(hx - r, -hy)),
        Part.Arc(v(hx - r, -hy), v(hx - r + r / sqrt2, -hy + r - r / sqrt2), v(hx, -hy + r)).toShape(),
        Part.makeLine(v(hx, -hy + r), v(hx, hy - r)),
        Part.Arc(v(hx, hy - r), v(hx - r + r / sqrt2, hy - r + r / sqrt2), v(hx - r, hy)).toShape(),
        Part.makeLine(v(hx - r, hy), v(-hx + r, hy)),
        Part.Arc(v(-hx + r, hy), v(-hx + r - r / sqrt2, hy - r + r / sqrt2), v(-hx, hy - r)).toShape(),
        Part.makeLine(v(-hx, hy - r), v(-hx, -hy + r)),
        Part.Arc(v(-hx, -hy + r), v(-hx + r - r / sqrt2, -hy + r - r / sqrt2), v(-hx + r, -hy)).toShape(),
    ]
    return Part.Face(Part.Wire(edges))


def build(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    width, length, height = 71.52, 146.71, 7.64
    opening_width, opening_length = 69.52, 144.71
    # No corner-radius callout was used: 8.0 mm is an explicitly provisional
    # visual assumption, not a value recovered from the drawing.
    outer_radius, inner_radius = 8.0, 7.0
    doc = App.newDocument('817_07781_A_outer_envelope_draft')
    outer = rounded_rectangle(width, length, outer_radius).extrude(App.Vector(0, 0, height))
    opening = rounded_rectangle(opening_width, opening_length, inner_radius, -0.1).extrude(App.Vector(0, 0, height + 0.2))
    shape = outer.cut(opening)
    if shape.isNull() or not shape.isValid() or not shape.Solids:
        raise RuntimeError('Draft frame failed B-Rep validity checks')
    obj = doc.addObject('Part::Feature', 'OuterEnvelopeDraft')
    obj.Label = '817-07781-A outer envelope draft - NOT MASTER CAD'
    obj.Shape = shape
    obj.addProperty('App::PropertyString', 'SourceDrawing', 'Reconstruction')
    obj.SourceDrawing = '817-07781-A.pdf, sheet 2'
    obj.addProperty('App::PropertyString', 'Status', 'Reconstruction')
    obj.Status = 'Approximate display geometry; not suitable for dimensional inspection'
    doc.recompute()
    step_path = output / '817-07781-A_outer-envelope_draft.step'
    fcstd_path = output / '817-07781-A_outer-envelope_draft.FCStd'
    Part.export([obj], str(step_path))
    doc.saveAs(str(fcstd_path))
    App.closeDocument(doc.Name)
    report = {
        'drawing_no': '817-07781-A',
        'artifact_kind': 'approximate outer-envelope CAD draft',
        'files': [step_path.name, fcstd_path.name],
        'explicit_dimensions_mm': {
            'overall_width': width,
            'overall_length': length,
            'cg_opening_width': opening_width,
            'cg_opening_length': opening_length,
            'band_thickness_interpreted_as_extrusion_height': height,
        },
        'assumptions_mm': {'outer_corner_radius': outer_radius, 'inner_corner_radius': inner_radius},
        'excluded': [
            'all internal ribs, bosses, threads, clips, trims, seams and local steps',
            'separate BOM components and all 11 threaded nuts',
            'drawing-specific free-state shape, local datums and FAI point coordinates',
        ],
        'authority': 'visual prototype only; drawing notes identify master geometric data as controlling',
        'brep_valid': bool(shape.isValid()),
        'solid_count': len(shape.Solids),
        'bbox_mm': [shape.BoundBox.XMin, shape.BoundBox.YMin, shape.BoundBox.ZMin,
                    shape.BoundBox.XMax, shape.BoundBox.YMax, shape.BoundBox.ZMax],
        'volume_mm3': shape.Volume,
    }
    (output / 'reconstruction_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/cyg/Oracle/reports/oracle-viewer/reconstruction/817-07781-A')
    build(out)
