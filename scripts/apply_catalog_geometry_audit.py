#!/usr/bin/env python3
"""Attach verified FreeCAD B-Rep results to catalog assets and item envelopes."""
from __future__ import annotations

import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAT = ROOT / "oracle_viewer/catalog"
SUPPORTED = {".step", ".stp", ".igs", ".iges"}


def main() -> None:
    hardware_path = CAT / "hardware.json"
    manifest_path = CAT / "manifest.json"
    hardware = json.loads(hardware_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads((CAT / "cad_renderability_audit.json").read_text())
    by_name = {row["file"]: row for row in audit.values()}
    status_counts: collections.Counter[str] = collections.Counter()
    unique_status_counts = collections.Counter(row.get("status", "error") for row in audit.values())
    item_counts: collections.Counter[str] = collections.Counter()
    missing = []

    for asset in manifest:
        row = by_name.get(asset["file"])
        if not row or pathlib.Path(asset.get("file", "")).suffix.lower() not in SUPPORTED:
            continue
        status = row.get("status", "error")
        asset["brep_validation"] = {"renderable": "valid", "renderable_derived_shape_fixed": "valid", "invalid_brep": "invalid", "no_faces": "invalid", "timed_out": "unresolved"}.get(status, "error")
        asset["bbox_mm"] = row.get("bbox_mm")
        asset["body_count"] = row.get("body_count")
        asset["face_count"] = row.get("face_count")
        asset["geometry_check"] = "FreeCAD Part.read + Shape.isValid"
        status_counts[status] += 1

    for item in hardware["items"]:
        preview_only = set(item.get("cad_preview_only_files", []))
        cad_names = [
            name for name in item.get("cad_files", [])
            if pathlib.Path(name).suffix.lower() in SUPPORTED and name not in preview_only
        ]
        rows = [by_name[name] for name in cad_names if name in by_name]
        if cad_names and not rows:
            missing.append(item["id"])
            item["cad_brep_validation"] = "pending"
            continue
        if not rows:
            item["cad_brep_validation"] = "unavailable"
            continue
        # A valid exact source model takes precedence over a sibling IGES timeout
        # or failed export. If none passes, preserve the distinction between an
        # invalid B-Rep, a worker timeout, and a reader error.
        primary = next((row for row in rows if row.get("status") in {"renderable", "renderable_derived_shape_fixed"}), rows[0])
        statuses = {row.get("status", "error") for row in rows}
        status = primary.get("status", "error")
        if statuses & {"renderable", "renderable_derived_shape_fixed"}:
            item_status = "valid"
        elif "timed_out" in statuses:
            item_status = "unresolved"
        elif "error" in statuses:
            item_status = "error"
        else:
            item_status = "invalid"
        item["cad_brep_validation"] = item_status
        item["cad_geometry_check"] = "FreeCAD Part.read + Shape.isValid"
        item["cad_bbox_mm"] = primary.get("bbox_mm")
        spec = item.setdefault("spec", {})
        field_source = item.setdefault("field_source", {})
        spec.pop("body_mm", None)
        field_source.pop("body_mm", None)
        if status in {"renderable", "renderable_derived_shape_fixed"} and primary.get("bbox_mm"):
            body = list(primary["bbox_mm"])
            unknown = item.setdefault("unknown_fields", [])
            published = spec.get("manufacturer_outer_dimensions_mm")
            dimension_conflict = False
            if isinstance(published, list) and len(published) == 3:
                actual_sorted = sorted(float(x) for x in body)
                published_sorted = sorted(float(x) for x in published)
                differences = [round(a - p, 4) for a, p in zip(actual_sorted, published_sorted)]
                conflict = any(abs(d) > max(2.0, 0.05 * p) for d, p in zip(differences, published_sorted))
                dimension_conflict = conflict
                item["cad_dimension_check"] = "published_dimension_mismatch_review" if conflict else "published_dimensions_match_within_tolerance"
                item["cad_dimension_comparison_mm_sorted"] = {"cad": actual_sorted, "manufacturer_catalog": published_sorted, "delta_cad_minus_catalog": differences}
                if conflict:
                    note = "CAD包围尺寸与官方手册外形尺寸不一致，已保留两组值；安装净空按较大包络复核。"
                    if note not in item.get("review_status", ""):
                        item["review_status"] = (item.get("review_status", "") + " " + note).strip()
                    if note not in unknown:
                        unknown.append(note)
            else:
                item["cad_dimension_check"] = "no_comparable_published_3d_dimensions"
            if dimension_conflict:
                # Use the larger major/middle/minor extents from both independent
                # sources. This can reject some feasible placements, but cannot
                # silently shrink the model for clearance screening.
                envelope = [round(max(float(c), float(p)), 4) for c, p in zip(body, published)]
                spec["body_mm"] = envelope
                spec["body_envelope_status"] = "conservative_max_of_validated_CAD_AABB_and_catalog_dimensions"
                field_source["body_mm"] = "conservative component-wise maximum in source axes of validated STEP AABB and manufacturer catalog outer dimensions"
                if "verified STEP physical envelope" in unknown:
                    unknown.remove("verified STEP physical envelope")
                note = "CAD与手册外形尺寸冲突；净空检查使用两者逐轴较大的保守包络，定版前复核安装方向。"
                if note not in unknown:
                    unknown.append(note)
                item_counts["conservative_dimension_conflict_envelope"] += 1
            else:
                spec["body_mm"] = body
                field_source["body_mm"] = f"validated STEP/STP/IGES AABB: {primary['file']}"
                spec["body_envelope_status"] = "validated_STEP_AABB"
                if "verified STEP physical envelope" in unknown:
                    unknown.remove("verified STEP physical envelope")
                item_counts["valid_body_envelope"] += 1
        else:
            spec.pop("body_mm", None)
            field_source.pop("body_mm", None)
            item_counts["no_valid_body_envelope"] += 1
        item_counts[item["cad_brep_validation"]] += 1

    # Geometry processing must preserve the catalog revision established by
    # the ingestion step that changed the source records.
    hardware_path.write_text(json.dumps(hardware, ensure_ascii=False, indent=2) + "\n")
    manifest_path.write_text(json.dumps(sorted(manifest, key=lambda row: (row.get("item_id", ""), row["file"], row.get("kind", ""))), ensure_ascii=False, indent=2) + "\n")

    gu_rows = {
        item["source_model"]: {
            "item_id": item["id"],
            "status": item.get("cad_brep_validation"),
            "file": item.get("cad_files", [None])[0] if item.get("cad_files") else None,
            "bbox_mm": item.get("cad_bbox_mm"),
            "manufacturer_outer_dimensions_mm": item.get("spec", {}).get("manufacturer_outer_dimensions_mm"),
            "dimension_check": item.get("cad_dimension_check"),
        }
        for item in hardware["items"]
        if item.get("vendor", "").startswith("GU Optics")
    }
    (CAT / "gu_optics_geometry_audit.json").write_text(json.dumps({"catalog_revision": hardware["revision"], "summary": dict(collections.Counter(row.get("status", "error") for row in gu_rows.values())), "items": gu_rows, "audit_missing": missing}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"revision": hardware["revision"], "unique_audited_cad_files": len(by_name), "unique_asset_statuses": dict(unique_status_counts), "manifest_asset_links_checked": sum(status_counts.values()), "catalog_item_statuses": dict(item_counts), "missing_audit_rows": missing[:20], "GU_records": len(gu_rows), "GU_geometry_counts": dict(collections.Counter(row["status"] for row in gu_rows.values())), "GU_dimension_mismatches": sum(row["dimension_check"] == "published_dimension_mismatch_review" for row in gu_rows.values())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
