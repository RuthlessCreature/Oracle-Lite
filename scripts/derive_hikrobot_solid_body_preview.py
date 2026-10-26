#!/usr/bin/env python3
"""Export valid closed solids from a HIKROBOT CAD source as a separate preview.

This never overwrites vendor CAD. It is intended for source files whose full
shape includes scan-volume/helper geometry or invalid non-solid surfaces. The
result is a derived, body-only preview and must not be used as a metrology
model or assumed to include all optical/measurement geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.append("/usr/lib/freecad-python3/lib")
import FreeCAD
import Part


def dims(shape):
    b = shape.BoundBox
    return [float(b.XLength), float(b.YLength), float(b.ZLength)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("target", type=Path)
    ap.add_argument("--min-solids", type=int, default=10,
                    help="Refuse thin range planes or small decorative solids")
    args = ap.parse_args()
    source = Part.read(str(args.source))
    solids = list(source.Solids)
    valid = [s for s in solids if s.isValid()]
    if len(valid) < args.min_solids:
        raise SystemExit(f"Only {len(valid)} valid solids; minimum is {args.min_solids}")
    body = Part.makeCompound(valid)
    if body.isNull() or not body.isValid():
        raise SystemExit("Filtered solid-body compound failed B-Rep validation")
    args.target.parent.mkdir(parents=True, exist_ok=True)
    doc = FreeCAD.newDocument("HikrobotSolidBodyPreview")
    try:
        obj = doc.addObject("Part::Feature", "SolidBodies")
        obj.Shape = body
        doc.recompute()
        Part.export([obj], str(args.target))
    finally:
        FreeCAD.closeDocument(doc.Name)
    check = Part.read(str(args.target))
    if check.isNull() or not check.isValid():
        args.target.unlink(missing_ok=True)
        raise SystemExit("Exported preview did not pass FreeCAD round-trip validation")
    if any(abs(a - b) > max(1e-6, abs(a) * 1e-8)
           for a, b in zip(dims(body), dims(check))):
        args.target.unlink(missing_ok=True)
        raise SystemExit("Exported preview envelope changed during round-trip")
    print(json.dumps({
        "source": str(args.source), "source_sha256": sha(args.source),
        "source_brep_valid": bool(source.isValid()),
        "source_face_count": len(source.Faces), "source_solid_count": len(solids),
        "retained_valid_solid_count": len(valid),
        "excluded_invalid_solid_count": len(solids) - len(valid),
        "source_full_shape_bbox_mm": dims(source),
        "derived_body_bbox_mm": dims(check),
        "target": str(args.target), "target_sha256": sha(args.target),
        "derived_brep_valid": True,
        "derivation": "FreeCAD Part.makeCompound of individually valid closed source solids; non-solid source geometry and invalid solids omitted",
        "scope": "body-only preview; not metrology CAD; measurement-volume, optical or annotation geometry may be omitted"
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
