#!/usr/bin/env python3
"""Persist SKU-level provenance and outcome for the GU Optics STEP batch."""
from __future__ import annotations

import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAT = ROOT / "oracle_viewer/catalog"
INDEX = pathlib.Path("/tmp/gu-light-library.json")
PROGRESS = pathlib.Path("/tmp/gu-optics-step-download-progress.json")
PROBES = CAT / "gu_optics_failed_step_probe.json"
OUT = CAT / "gu_optics_download_audit.json"


def main() -> None:
    rows = json.loads(INDEX.read_text())["rows"]
    outcomes = json.loads(PROGRESS.read_text())
    probe_doc = json.loads(PROBES.read_text()) if PROBES.exists() else {"items": []}
    probes = {row["model"]: row for row in probe_doc["items"]}
    entries = []
    for row in rows:
        result = outcomes.get(row["model"], {"status": "not_attempted"})
        probe = probes.get(row["model"])
        if result.get("status") == "downloaded":
            status = "downloaded"
        elif probe and (probe.get("page_title") or "").startswith("404"):
            status = "source_404_html"
        elif probe and probe.get("http_status") == 200 and probe.get("received_prefix_bytes") == 0:
            status = "source_empty_response"
        else:
            status = result.get("status")
        entries.append(
            {
                "model": row["model"],
                "title": row.get("name"),
                "category": row.get("category_kind"),
                "source_page": row.get("source_detail"),
                "advertised_step_url": result.get("url")
                or ("https://www.gu-optics.com" + row["step_path"] if row.get("step_path") else None),
                "status": status,
                "download_attempt_status": result.get("status"),
                "source_probe": probe,
                "local_asset": f"assets/{result['file']}" if result.get("file") else None,
                "sha256": result.get("sha256"),
                "bytes": result.get("bytes"),
                "retrieved_at": result.get("retrieved_at"),
                "failure_detail": result.get("sample") or result.get("error"),
            }
        )
    counts = collections.Counter(entry["status"] for entry in entries)
    unique_assets = {entry["local_asset"] for entry in entries if entry["local_asset"]}
    unique_asset_bytes = sum(
        (CAT / asset).stat().st_size for asset in unique_assets if (CAT / asset).is_file()
    )
    data = {
        "schema_version": 1,
        "source": "GU Optics Shanghai public machine-vision lighting product catalog; reseller listing, OEM identity unverified",
        "catalog_revision": "2026-10-14.34",
        "summary": {
            "listed_skus": len(entries),
            "status_counts": dict(sorted(counts.items())),
            "unique_downloaded_step_assets": len(unique_assets),
            "unique_downloaded_step_bytes": unique_asset_bytes,
            "downloaded_step_bytes_by_sku": sum(entry["bytes"] or 0 for entry in entries),
        },
        "items": entries,
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(data["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
