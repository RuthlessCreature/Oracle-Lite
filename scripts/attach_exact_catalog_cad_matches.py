"""Attach existing source CAD assets to expansion SKUs by exact model label.

Matching is limited to punctuation/case normalization and identical component
kind. No family-level or prefix-only CAD reuse is allowed here.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "oracle_viewer" / "catalog"


def norm(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def main():
    hp, mp = CAT / "hardware.json", CAT / "manifest.json"
    bp = CAT / "batches" / "hardware-expansion-2000.json"
    hw, manifest, batch = json.loads(hp.read_text()), json.loads(mp.read_text()), json.loads(bp.read_text())
    targets = {x["id"]: x for x in hw["items"] if x["id"] in set(batch["added_ids"])}
    source_rows = {x["id"]: x for x in hw["items"]}
    # Only model identity fields that a source catalog explicitly labels.
    aliases = collections.defaultdict(list)
    for row in hw["items"]:
        if not row.get("cad_files"):
            continue
        for field in ("id", "source_model", "source_model_label", "manufacturer_model", "cad_model_ref"):
            value = row.get(field)
            if value:
                aliases[(norm(value), row.get("kind"))].append((row["id"], field, str(value), row["cad_files"]))
    manifest_by_file = collections.defaultdict(list)
    for asset in manifest:
        if asset.get("file"):
            manifest_by_file[asset["file"]].append(asset)
    new_links, evidence = [], []
    for target in targets.values():
        candidates = aliases.get((norm(target["id"]), target.get("kind")), [])
        candidates = [x for x in candidates if x[0] != target["id"]]
        # Deduplicate repeated item/manifest aliases of the same source file.
        unique = {}
        for source_id, field, label, files in candidates:
            for filename in files:
                if (CAT / "assets" / filename).is_file():
                    unique.setdefault((source_id, filename), (source_id, field, label, filename))
        source_groups = collections.defaultdict(list)
        for candidate in unique.values():
            source_groups[candidate[0]].append(candidate)
        if len(source_groups) > 1:
            evidence.append({"target_id": target["id"], "status": "ambiguous_multiple_exact_source_items", "candidates": list(unique.values())})
            continue
        if not unique:
            continue
        source_id = next(iter(source_groups))
        candidates_for_source = source_groups[source_id]
        # A single source record may provide several neutral formats for one SKU;
        # these are complementary files, not an identity conflict.
        label = candidates_for_source[0][2]
        filenames = [x[3] for x in candidates_for_source if Path(x[3]).suffix.lower() in {".step", ".stp", ".igs", ".iges"}]
        if not filenames:
            continue
        for filename in filenames:
            if filename not in target.setdefault("cad_files", []):
                target["cad_files"].append(filename)
        target["cad_model_ref"] = label
        target["cad_status"] = "exact_source_sku_cad_attached_geometry_pending_validation"
        target["cad_scope"] = "Source catalog explicitly labels this CAD with the same model code after case/punctuation normalization. Exact variant geometry and installation envelope still require B-Rep and dimensional review."
        target["cad_matching_method"] = "exact_model_code_normalized_case_and_punctuation_only"
        target["cad_source_item_id"] = source_id
        target["cad_brep_validation"] = "pending"
        target["review_status"] = "Exact source-model CAD linked; CAD geometry and stated product envelope pending validation"
        target["layout_adapter_ready"] = False
        for filename in filenames:
            source_assets = manifest_by_file[filename]
            source_asset = next((a for a in source_assets if a.get("item_id") == source_id), source_assets[0])
            new_manifest = dict(source_asset)
            new_manifest["item_id"] = target["id"]
            new_manifest["retrieved_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            new_manifest["cad_scope"] = target["cad_scope"]
            new_manifest["matched_source_item_id"] = source_id
            new_manifest["model_match"] = {"target_model": target["id"], "source_model": label, "normalization": "case-fold plus remove punctuation only", "component_kind": target["kind"]}
            if not any(a.get("item_id") == target["id"] and a.get("file") == filename for a in manifest):
                manifest.append(new_manifest)
            new_links.append({"target_id": target["id"], "source_id": source_id, "source_model_label": label, "file": filename, "format": Path(filename).suffix.lower().lstrip("."), "source": source_asset.get("source_page") or source_asset.get("url"), "sha256": source_asset.get("sha256"), "bytes": source_asset.get("bytes")})
    if new_links:
        rev = hw.get("revision", "2026-10-21.9")
        try:
            date, seq = rev.rsplit(".", 1)
            hw["revision"] = f"{date}.{int(seq) + 1}"
        except (ValueError, TypeError):
            hw["revision"] = "2026-10-21.10"
        hw["items"] = sorted(hw["items"], key=lambda x: (x.get("kind", ""), x.get("vendor", ""), x["id"]))
        hp.write_text(json.dumps(hw, ensure_ascii=False, indent=2) + "\n")
        mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    batch["cad_exact_match_links"] = new_links
    batch["cad_exact_match_count"] = len(new_links)
    batch["cad_exact_match_policy"] = "Exact model-code match after case/punctuation normalization and same component kind; no series/prefix inference. CAD geometry remains pending review."
    batch["cad_exact_match_audit"] = evidence
    bp.write_text(json.dumps(batch, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"attached": len(new_links), "files": new_links, "ambiguous": evidence}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
