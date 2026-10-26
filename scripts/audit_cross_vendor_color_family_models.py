#!/usr/bin/env python3
"""Find downloadable CAD whose neutral family code matches a missing OPT SKU.

Audit only. It never attaches a cross-vendor file automatically because an
identical base code does not prove OEM identity, revision, or color-specific
housing. Output can support a clearly labeled family preview after review.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "oracle_viewer" / "catalog"
ASSETS = CATALOG / "assets"
OUTPUT = CATALOG / "cross_vendor_color_family_model_audit.json"
CAD_SUFFIXES = {".step", ".stp", ".igs", ".iges"}
COLOR_SUFFIX = re.compile(r"-(?P<color>[BGRW])$", re.I)


def strings_for_asset(item: dict, entry: dict) -> list[str]:
    values = [
        item.get("id"), item.get("source_model"), item.get("source_model_label"),
        entry.get("manufacturer_file_name"), entry.get("source_model_label"),
        entry.get("source_model_code"),
    ]
    return [str(value) for value in values if value]


def base_code(item: dict) -> tuple[str, str | None]:
    model = item.get("spec", {}).get("manufacturer_parameters", {}).get("Model") or item["id"]
    model = str(model)
    if model.upper().startswith("OPT-"):
        model = model[4:]
    match = COLOR_SUFFIX.search(model)
    color = match.group("color").upper() if match else None
    if match:
        model = model[:match.start()]
    return model, color


def main() -> None:
    hardware = json.loads((CATALOG / "hardware.json").read_text())
    manifest = json.loads((CATALOG / "manifest.json").read_text())
    renderability_path = CATALOG / "cad_renderability_audit.json"
    renderability = json.loads(renderability_path.read_text()) if renderability_path.exists() else {}
    by_item: dict[str, list[dict]] = {}
    for row in manifest:
        by_item.setdefault(row.get("item_id", ""), []).append(row)

    candidates = []
    for target in hardware["items"]:
        if target.get("vendor") != "OPT" or target.get("cad_files"):
            continue
        code, color = base_code(target)
        pattern = re.compile(rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])", re.I)
        for source in hardware["items"]:
            if source.get("vendor") == "OPT" or not source.get("cad_files"):
                continue
            for entry in by_item.get(source["id"], []):
                filename = entry.get("file", "")
                if Path(filename).suffix.lower() not in CAD_SUFFIXES:
                    continue
                labels = strings_for_asset(source, entry)
                matching = [label for label in labels if pattern.search(label)]
                if not matching:
                    continue
                audit = renderability.get(filename, {})
                candidates.append({
                    "target_item_id": target["id"],
                    "target_base_model_code": code,
                    "target_color_suffix": color,
                    "target_source_url": target.get("source_url"),
                    "source_item_id": source["id"],
                    "source_vendor": source.get("vendor"),
                    "source_model": source.get("source_model"),
                    "source_model_label": entry.get("source_model_label") or source.get("source_model_label"),
                    "source_matching_labels": matching,
                    "source_page": entry.get("source_page") or source.get("source_url"),
                    "file": filename,
                    "sha256": entry.get("sha256") or hashlib.sha256((ASSETS / filename).read_bytes()).hexdigest(),
                    "format": Path(filename).suffix.lower(),
                    "brep_audit_status": audit.get("status", "not_audited"),
                    "brep_valid": audit.get("brep_valid"),
                    "bbox_mm": audit.get("bbox_mm"),
                    "body_count": audit.get("body_count"),
                    "face_count": audit.get("face_count"),
                    "assessment": "exact base model code match; source color neutral/unspecified; same OEM, revision, and target-SKU geometry remain unverified",
                })

    result = {
        "catalog_revision": hardware.get("revision"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Exact case-insensitive model-code token match after removing OPT- vendor prefix and optional terminal B/G/R/W color suffix; compare only assets attached to non-OPT catalog items.",
        "automatic_binding": False,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("catalog_revision", "candidate_count", "automatic_binding")}, ensure_ascii=False))
    for row in candidates:
        print(row["target_item_id"], "<=", row["source_item_id"], row["file"], row["brep_valid"])


if __name__ == "__main__":
    main()
