#!/usr/bin/env python3
"""Find exact CAD geometry shared by color-coded sibling SKUs.

This is an audit only: it does not change hardware.json or manifest.json.
Every available color sibling is fingerprinted using its complete B-Rep
geometry. Missing SKUs are candidates only when all available siblings have
the same solid, face, edge, and vertex signature.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "oracle_viewer" / "catalog"
ASSETS = CATALOG / "assets"
OUTPUT = CATALOG / "color_variant_geometry_audit.json"
COLOR_SUFFIX = re.compile(r"^(?P<family>.+)-(?P<color>[BGRW])$", re.I)
PRECISION_DIGITS = 5


def q(value: float) -> float:
    return round(float(value), PRECISION_DIGITS)


def xyz(point) -> tuple[float, float, float]:
    return tuple(q(getattr(point, axis)()) for axis in ("X", "Y", "Z"))


def bbox(shape) -> tuple[float, ...]:
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    return tuple(q(value) for value in box.Get())


def properties(shape, mode: str) -> tuple[float, tuple[float, float, float]]:
    result = GProp_GProps()
    if mode == "face":
        BRepGProp.SurfaceProperties_s(shape, result)
    elif mode == "edge":
        BRepGProp.LinearProperties_s(shape, result)
    else:
        BRepGProp.VolumeProperties_s(shape, result)
    return q(result.Mass()), xyz(result.CentreOfMass())


def count_subshapes(shape, kind: int) -> int:
    explorer = TopExp_Explorer(shape, kind)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def multiset_signature(rows: list[tuple]) -> list[list[object]]:
    return [[list(signature), count] for signature, count in sorted(Counter(rows).items())]


def geometry_fingerprint(filename: str) -> dict[str, object]:
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(ASSETS / filename))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise ValueError(f"STEP read failed: {filename}")
    if reader.TransferRoots() < 1:
        raise ValueError(f"STEP transfer failed: {filename}")
    shape = reader.OneShape()
    if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError(f"invalid B-Rep: {filename}")

    rows: dict[str, list[tuple]] = {"faces": [], "edges": [], "vertices": []}
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        area, center = properties(face, "face")
        adaptor = BRepAdaptor_Surface(face, True)
        surface = str(adaptor.GetType())
        parameters: tuple[float, ...] = ()
        if "Cylinder" in surface:
            parameters = (q(adaptor.Cylinder().Radius()),)
        elif "Cone" in surface:
            cone = adaptor.Cone()
            parameters = (q(cone.RefRadius()), q(cone.SemiAngle()))
        rows["faces"].append((surface, parameters, area, center, bbox(face)))
        explorer.Next()

    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        length, center = properties(edge, "edge")
        curve = str(BRepAdaptor_Curve(edge).GetType())
        rows["edges"].append((curve, length, center, bbox(edge)))
        explorer.Next()

    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while explorer.More():
        vertex = TopoDS.Vertex_s(explorer.Current())
        rows["vertices"].append(xyz(BRep_Tool.Pnt_s(vertex)))
        explorer.Next()

    volume, center = properties(shape, "solid")
    solid_count = count_subshapes(shape, TopAbs_SOLID)
    envelope = bbox(shape)
    signatures = {name: multiset_signature(items) for name, items in rows.items()}
    canonical = json.dumps(
        {"shape": [volume, center, envelope, solid_count], **signatures},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "file": filename,
        "sha256": hashlib.sha256((ASSETS / filename).read_bytes()).hexdigest(),
        "geometry_signature_sha256": hashlib.sha256(canonical).hexdigest(),
        "brep_valid": True,
        "solid_count": solid_count,
        "face_count": len(rows["faces"]),
        "edge_count": len(rows["edges"]),
        "vertex_count": len(rows["vertices"]),
        "volume_mm3": volume,
        "center_of_mass_mm": center,
        "bbox_mm": [q(envelope[i + 3] - envelope[i]) for i in range(3)],
        "face_signature": signatures["faces"],
        "edge_signature": signatures["edges"],
        "vertex_signature": signatures["vertices"],
    }


def main() -> None:
    hardware = json.loads((CATALOG / "hardware.json").read_text())
    previous_families: set[str] = set()
    if OUTPUT.exists():
        try:
            prior = json.loads(OUTPUT.read_text())
            previous_families = {row["family_id"] for row in prior.get("families", [])}
        except (OSError, ValueError, KeyError, TypeError):
            previous_families = set()
    groups: dict[str, list[dict[str, object]]] = {}
    for item in hardware["items"]:
        match = COLOR_SUFFIX.match(item.get("id", ""))
        if match:
            groups.setdefault(match.group("family"), []).append(item)

    fingerprint_cache: dict[str, dict[str, object]] = {}
    candidates = []
    for family, siblings in sorted(groups.items()):
        missing = [item for item in siblings if not item.get("cad_files")]
        available = [item for item in siblings if item.get("cad_files")]
        if len(available) < 2 or (not missing and family not in previous_families):
            continue
        sibling_models = []
        for item in available:
            model_fingerprints = []
            for filename in item["cad_files"]:
                if Path(filename).suffix.lower() not in {".step", ".stp"}:
                    continue
                try:
                    fingerprint_cache.setdefault(filename, geometry_fingerprint(filename))
                    model_fingerprints.append(fingerprint_cache[filename])
                except Exception as exc:  # Keep each failure visible in the audit.
                    model_fingerprints.append({"file": filename, "error": str(exc)})
            sibling_models.append({"item_id": item["id"], "models": model_fingerprints})
        valid = [model for row in sibling_models for model in row["models"] if model.get("brep_valid")]
        hashes = {model["geometry_signature_sha256"] for model in valid}
        if len(valid) < 2:
            status = "insufficient_valid_sibling_models"
        elif len(hashes) == 1 and len(valid) == sum(len(row["models"]) for row in sibling_models):
            status = "identical_complete_geometric_signatures"
        else:
            status = "sibling_geometry_differs_or_unresolved"
        candidates.append(
            {
                "family_id": family,
                "status": status,
                "missing_sku_ids": [item["id"] for item in missing],
                "available_sku_ids": [item["id"] for item in available],
                "available_sibling_models": sibling_models,
                "signature_precision_mm": 10 ** -PRECISION_DIGITS,
                "method": "OpenCascade valid B-Rep; compare solid metrics and sorted multisets of face surface/area/center/bbox, edge curve/length/center/bbox, and vertex coordinates",
                "note": "Geometric identity supports family-shape reuse for layout. It does not establish color appearance, OEM identity, revision, or target-SKU-specific manufacturing differences.",
            }
        )

    result = {
        "catalog_revision": hardware.get("revision"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": Path(__file__).name,
        "candidate_family_count": len(candidates),
        "families_with_identical_geometry": sum(
            item["status"] == "identical_complete_geometric_signatures" for item in candidates
        ),
        "families": candidates,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("catalog_revision", "candidate_family_count", "families_with_identical_geometry")}, ensure_ascii=False))
    for family in candidates:
        print(family["family_id"], family["status"], "missing=", family["missing_sku_ids"], "available=", family["available_sku_ids"])


if __name__ == "__main__":
    main()
