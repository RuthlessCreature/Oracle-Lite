"""Run FreeCAD B-Rep checks for exact CAD assets linked to the 2,000-SKU batch."""
from __future__ import annotations

import collections
import concurrent.futures
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAT = ROOT / "oracle_viewer" / "catalog"
WORKER = ROOT / "scripts" / "cad_renderability_worker.py"


def validate(filename: str) -> dict:
    path = CAT / "assets" / filename
    # IGES files may contain very large, unhealed surface assemblies and can
    # take considerably longer than STEP solids to import in FreeCAD.
    timeout = 120 if path.suffix.lower() in {".igs", ".iges"} else 60
    try:
        process = subprocess.run([sys.executable, str(WORKER), str(path)], capture_output=True, text=True, timeout=timeout)
        for line in reversed(process.stdout.splitlines()):
            if line.startswith("ORACLE_RESULT "):
                return json.loads(line[len("ORACLE_RESULT "):])
        return {"file": filename, "status": "error", "error": (process.stderr or process.stdout)[-1000:]}
    except subprocess.TimeoutExpired:
        return {"file": filename, "status": "timed_out", "error": f"FreeCAD B-Rep import exceeded {timeout} seconds"}


def main():
    hp, mp, bp = CAT / "hardware.json", CAT / "manifest.json", CAT / "batches" / "hardware-expansion-2000.json"
    hw, manifest, batch = json.loads(hp.read_text()), json.loads(mp.read_text()), json.loads(bp.read_text())
    targets = set(batch["added_ids"])
    items = {x["id"]: x for x in hw["items"]}
    filenames = sorted({f for item_id in targets for f in items[item_id].get("cad_files", [])
                        if pathlib.Path(f).suffix.lower() in {".step", ".stp", ".igs", ".iges"}})
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(validate, filenames))
    by_file = {x["file"]: x for x in results}
    now = datetime.now(timezone.utc).isoformat()
    for item_id in targets:
        item = items[item_id]
        refs = [by_file[f] for f in item.get("cad_files", []) if f in by_file]
        if not refs:
            continue
        item["cad_geometry_validation"] = {x["file"]: {k: v for k, v in x.items() if k != "file"} for x in refs}
        states = {x["status"] for x in refs}
        # A SKU is renderable if at least one exact-SKU neutral file imports as
        # a valid B-Rep. Preserve per-file failures above for auditability.
        item["cad_brep_validation"] = (
            "renderable" if "renderable" in states else
            "invalid_brep_review_required" if "invalid_brep" in states else
            ";".join(sorted(states))
        )
        item["layout_adapter_ready"] = False
        item["cad_validation_at"] = now
    result_by_name = {x["file"]: x for x in results}
    for asset in manifest:
        if asset.get("file") in result_by_name and asset.get("item_id") in targets:
            row = result_by_name[asset["file"]]
            asset["geometry_validation"] = {k: v for k, v in row.items() if k != "file"}
            asset["geometry_validation_at"] = now
    batch["cad_exact_match_file_count"] = len(batch.get("cad_exact_match_links", []))
    batch["cad_exact_match_sku_count"] = len({x["target_id"] for x in batch.get("cad_exact_match_links", [])})
    batch["cad_exact_match_geometry_status_counts"] = dict(collections.Counter(x["status"] for x in results))
    batch["cad_exact_match_geometry_validation"] = results
    batch["cad_exact_match_geometry_validated_at"] = now
    batch["model_specific_cad_coverage"] = {"matched_skus": len({x["target_id"] for x in batch.get("cad_exact_match_links", [])}), "target_skus": len(targets), "fraction": round(len({x["target_id"] for x in batch.get("cad_exact_match_links", [])}) / max(1, len(targets)), 6), "note": "Only exact model-code source matches are counted; all geometry links remain unavailable for collision-approved placement until envelope/orientation validation."}
    hw["items"] = sorted(hw["items"], key=lambda x: (x.get("kind", ""), x.get("vendor", ""), x["id"]))
    hp.write_text(json.dumps(hw, ensure_ascii=False, indent=2) + "\n")
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    bp.write_text(json.dumps(batch, ensure_ascii=False, indent=2) + "\n")
    report = {"title": "Exact-source CAD coverage audit for 2,000-device expansion", "created_at": now,
              "target_skus": len(targets), "matched_skus": batch["cad_exact_match_sku_count"],
              "matched_sku_fraction": batch["model_specific_cad_coverage"]["fraction"],
              "neutral_cad_files": len(results), "geometry_status_counts": batch["cad_exact_match_geometry_status_counts"],
              "files": results, "layout_adapter_ready_items": 0,
              "limitations": ["The manufacturer/distributor model string is an exact SKU match after punctuation/case normalization.", "FreeCAD B-Rep validity is not equivalent to dimensional or mounting-datum validation.", "No model is enabled for collision-checked placement solely by this import."]}
    (CAT / "cad_expansion_2000_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"target_skus": len(targets), "matched_skus": batch["cad_exact_match_sku_count"], "neutral_cad_files": len(results), "geometry_status_counts": batch["cad_exact_match_geometry_status_counts"], "layout_adapter_ready_items": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
