#!/usr/bin/env python3
"""Keep GU product listings in the library when their advertised STEP is unavailable."""
from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAT = ROOT / "oracle_viewer/catalog"
INDEX = pathlib.Path("/tmp/gu-light-library.json")
PROGRESS = pathlib.Path("/tmp/gu-optics-step-download-progress.json")


def main() -> None:
    rows = json.loads(INDEX.read_text())["rows"]
    progress = json.loads(PROGRESS.read_text())
    hardware_path = CAT / "hardware.json"
    hardware = json.loads(hardware_path.read_text())
    items = {item["id"]: item for item in hardware["items"]}
    added = 0
    for row in rows:
        result = progress.get(row["model"], {})
        if result.get("status") == "downloaded":
            continue
        sku = row["model"]
        item_id = "GU-" + re.sub(r"[^A-Za-z0-9-]+", "-", sku).strip("-").upper()
        if item_id in items:
            continue
        specs = {
            str(spec["title"]): spec.get("attr_value")
            for spec in row.get("specs") or []
            if isinstance(spec, dict) and spec.get("title")
        }
        try:
            price = float(str(row.get("price", "")).replace(",", "").replace("￥", "").strip())
        except (TypeError, ValueError):
            price = None
        availability = "；".join(str(value) for value in (row.get("availability"), row.get("warehouse")) if value)
        source = row["source_detail"]
        item = {
            "id": item_id,
            "kind": "controller" if row["category_kind"] == "controller" else "light",
            "vendor": "GU Optics Shanghai product listing (reseller; OEM identity unverified)",
            "source_url": source,
            "datasheet_url": "https://www.gu-optics.com" + row["pdf_path"] if row.get("pdf_path") else None,
            "spec": specs,
            "price": price,
            "price_currency": "CNY",
            "price_source": "GU Optics public product detail page; recheck before purchase",
            "availability": availability or "网页未注明，请向供方核实",
            "lifecycle_status": "listed_status_unverified",
            "cad_status": "source_step_link_unavailable",
            "review_status": "产品页列有 STEP 链接，但下载端返回 404 HTML 或空响应；未以相近型号替代 3D 模型。",
            "unknown_fields": ["精确3D模型", "OEM/制造商身份", "当前库存和交期", "机械安装尺寸与修订确认"],
            "field_source": {key: source for key in specs},
            "source_model": sku,
            "source_title": row.get("name"),
            "source_category": row.get("category_kind"),
            "source_supplier_code": row.get("supplier_model"),
            "source_model_id": row.get("source_id"),
            "source_vendor_status": "GU Optics Shanghai public catalog listing/reseller; OEM/manufacturer not verified",
            "cad_files": [],
            "source_step_url": result.get("url"),
            "cad_scope": "No solid CAD file was returned by the listed source link; do not use as a 3D component or for collision checks.",
            "source_pages": [source],
        }
        items[item_id] = item
        added += 1
    hardware["items"] = sorted(items.values(), key=lambda item: (item["kind"], item.get("vendor", ""), item["id"]))
    hardware["revision"] = "2026-10-14.34"
    hardware_path.write_text(json.dumps(hardware, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"revision": hardware["revision"], "unavailable_step_records_added": added, "catalog_total": len(items)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
