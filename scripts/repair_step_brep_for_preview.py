#!/usr/bin/env python3
"""Create a separate ShapeFix-repaired STEP copy without modifying the vendor original."""
import argparse
import hashlib
import json
from pathlib import Path
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.ShapeFix import ShapeFix_Shape
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer


def metrics(shape):
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    volume, area = GProp_GProps(), GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume)
    BRepGProp.SurfaceProperties_s(shape, area)
    return {"bbox": list(box.Get()), "volume_mm3": volume.Mass(), "area_mm2": area.Mass()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("target", type=Path)
    args = ap.parse_args()
    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.source)) != IFSelect_RetDone:
        raise SystemExit("Could not read source STEP")
    if reader.TransferRoots() == 0:
        raise SystemExit("STEP has no transferable roots")
    original = reader.OneShape()
    if original.IsNull():
        raise SystemExit("STEP produced a null shape")
    before = metrics(original)
    fixer = ShapeFix_Shape(original)
    fixer.SetPrecision(1e-5)
    fixer.SetMaxTolerance(1e-2)
    fixer.Perform()
    fixed = fixer.Shape()
    if fixed.IsNull() or not BRepCheck_Analyzer(fixed).IsValid():
        raise SystemExit("ShapeFix result is still not a valid B-Rep; source left unchanged")
    after = metrics(fixed)
    for key in ("bbox", "volume_mm3", "area_mm2"):
        a, b = before[key], after[key]
        av, bv = (a, b) if isinstance(a, list) else ([a], [b])
        if any(abs(x-y) > max(1e-8, abs(x)*1e-10) for x, y in zip(av, bv)):
            raise SystemExit(f"ShapeFix changed {key}; refusing to write repaired model")
    args.target.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    writer.Transfer(fixed, STEPControl_AsIs)
    if writer.Write(str(args.target)).name != "IFSelect_RetDone":
        args.target.unlink(missing_ok=True)
        raise SystemExit("Could not write repaired STEP")
    print(json.dumps({"source": str(args.source), "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      "target": str(args.target), "target_sha256": hashlib.sha256(args.target.read_bytes()).hexdigest(),
                      "repair": "OpenCascade ShapeFix_Shape; precision=1e-5 mm; max_tolerance=1e-2 mm",
                      "brep_valid": True, "geometry_metrics_unchanged": True,
                      "metrics": after}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
