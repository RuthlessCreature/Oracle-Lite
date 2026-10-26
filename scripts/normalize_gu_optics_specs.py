#!/usr/bin/env python3
"""Add source-traceable canonical optical/electrical fields to GU catalog records."""
from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG = ROOT / "oracle_viewer/catalog/hardware.json"
PROBE_FILE = ROOT / "oracle_viewer/catalog/gu_optics_failed_step_probe.json"
PDF_ENRICHMENT = ROOT / "oracle_viewer/catalog/gu_official_catalog_enrichment.json"

TYPE_MAP = {
    "ring": "ring",
    "bar": "bar",
    "uniform_bar": "bar",
    "combined_bar": "bar",
    "high_intensity_line": "bar",
    "backlight": "backlight",
    "parallel_backlight": "backlight",
    "side_backlight": "side_backlight",
    "hole_backlight": "hole_backlight",
    "coaxial": "coaxial",
    "dome": "dome",
    "spot": "spot",
    "ringless": "ringless",
    "squareless": "squareless",
    "telecentric": "telecentric",
    "nir": "near_infrared",
}
COLOR_MAP = {"B": "Blue", "G": "Green", "R": "Red", "W": "White"}
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def number(value: object) -> float | None:
    match = NUMBER.search(str(value or ""))
    return float(match.group()) if match else None


def main() -> None:
    data = json.loads(CATALOG.read_text())
    probes = {row["model"]: row for row in json.loads(PROBE_FILE.read_text()).get("items", [])} if PROBE_FILE.exists() else {}
    enrichment = json.loads(PDF_ENRICHMENT.read_text()) if PDF_ENRICHMENT.exists() else {}
    controller_rows = enrichment.get("controller_rows", {})
    changed = 0
    counts: dict[str, int] = {}
    for item in data["items"]:
        category = item.get("source_category")
        if item.get("vendor", "").startswith("GU Optics") is False or not category:
            continue
        spec = item.setdefault("spec", {})
        field_source = item.setdefault("field_source", {})
        if item.get("source_model") in probes:
            probe = probes[item["source_model"]]
            item["source_step_probe"] = probe
            if (probe.get("page_title") or "").startswith("404"):
                item["cad_status"] = "source_step_url_404"
                item["review_status"] = "商品页列有 STEP 链接；逐 SKU 复查返回 404 页面。未下载到 3D 文件，未用相似型号替代。"
            elif probe.get("http_status") == 200 and probe.get("received_prefix_bytes") == 0:
                item["cad_status"] = "source_step_empty_response"
                item["review_status"] = "商品页列有 STEP 链接；逐 SKU 复查返回 HTTP 200 但无文件内容。未下载到 3D 文件。"
        spec["source_category"] = category
        spec["type"] = TYPE_MAP.get(category, category)
        field_source["source_category"] = "source_product_category"
        field_source["type"] = "source_product_category_normalized"

        color = spec.get("颜色")
        if color:
            color = COLOR_MAP.get(str(color).strip().upper(), str(color).strip())
            spec["color"] = color
            field_source["color"] = "source_product_parameter:颜色"

        raw_power = next((v for k, v in spec.items() if "发光功率" in k or k.lower() == "power"), None)
        raw_voltage = next((v for k, v in spec.items() if "电压" in k or k.lower() == "voltage"), None)
        if raw_power is not None and (value := number(raw_power)) is not None:
            spec["power_w"] = value
            field_source["power_w"] = "source_product_parameter:发光功率"
        if raw_voltage is not None and (value := number(raw_voltage)) is not None:
            spec["voltage_v"] = value
            field_source["voltage_v"] = "source_product_parameter:电压"
        if spec.get("power_w") and spec.get("voltage_v"):
            spec["current_a"] = round(spec["power_w"] / spec["voltage_v"], 6)
            field_source["current_a"] = "derived_from_source_power_w_divided_by_voltage_v"

        if category == "controller":
            catalog_row = controller_rows.get(str(item.get("source_model")))
            if catalog_row:
                spec["manufacturer_catalog_version"] = "GU Optics Electronic Catalog 9.0"
                spec["manufacturer_catalog_pdf_page"] = catalog_row["pdf_page"]
                spec["manufacturer_catalog_row"] = catalog_row["row"]
                field_source["manufacturer_catalog_row"] = f"oracle_viewer/catalog/source_documents/gu-optics/gu-optics-electronic-catalog-9.0.pdf#pdf-page-{catalog_row['pdf_page']}"
                row_voltages = [float(value) for value in re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*V", catalog_row["row"], re.I)]
                if len(row_voltages) >= 2 and abs(row_voltages[0] - row_voltages[-1]) > 0.01:
                    conflict = {
                        "model": item.get("source_model"),
                        "field": "voltage_v",
                        "catalog_product_name_voltage_v": row_voltages[0],
                        "catalog_voltage_column_v": row_voltages[-1],
                        "product_detail_parameter_v": spec.get("voltage_v"),
                    }
                    item["source_conflicts"] = [*item.get("source_conflicts", []), conflict]
                    note = "官方总目录中产品名称电压与表格电压列不一致，控制器选型前必须向供方确认。"
                    if note not in item.get("review_status", ""):
                        item["review_status"] = (item.get("review_status", "") + " " + note).strip()
            channels = next((v for k, v in spec.items() if "通道" in k or k.lower() == "channels"), None)
            if (value := number(channels)) is not None:
                spec["channels"] = int(value)
                field_source["channels"] = "source_product_parameter:通道数"
            output_power = next((v for k, v in spec.items() if "输出功率" in k or k.lower() == "output_power_w"), None)
            if (value := number(output_power)) is not None:
                spec["output_power_w"] = value
                field_source["output_power_w"] = "source_product_parameter:输出功率"
            if spec.get("voltage_v") is not None:
                spec["output_voltage_v"] = spec["voltage_v"]
                field_source["output_voltage_v"] = field_source.get("voltage_v", "source_product_parameter:电压")
            if spec.get("output_power_w") and spec.get("output_voltage_v"):
                spec["output_current_a"] = round(spec["output_power_w"] / spec["output_voltage_v"], 6)
                field_source["output_current_a"] = "derived_from_source_output_power_w_divided_by_output_voltage_v"
            dimming_mode = next((v for k, v in spec.items() if "调光方式" in k), None)
            if dimming_mode:
                spec["dimming_mode"] = dimming_mode
                if "串口" in str(dimming_mode):
                    spec["interface"] = "serial + front-panel controls; protocol unspecified" if "按键" in str(dimming_mode) else "serial; protocol unspecified"
                else:
                    spec["interface"] = "manual only"
                field_source["dimming_mode"] = "source_product_parameter:调光方式"
                field_source["interface"] = "normalized_from_source_dimming_mode"

            title = str(item.get("source_title", ""))
            title_voltage = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*V", title, re.I)
            if title_voltage and spec.get("voltage_v") is not None and abs(float(title_voltage.group(1)) - float(spec["voltage_v"])) > 0.01:
                conflict = {"model": item.get("source_model"), "field": "voltage_v", "product_title": float(title_voltage.group(1)), "product_parameter": spec["voltage_v"]}
                item["source_conflicts"] = [*item.get("source_conflicts", []), conflict]
                note = "商品标题电压与参数字段不一致，控制器选型前必须向供方确认。"
                if note not in item.get("review_status", ""):
                    item["review_status"] = (item.get("review_status", "") + " " + note).strip()

        # Active area is populated only from explicitly named emitting-area/beam fields.
        active_source = next(
            ((k, v) for k, v in spec.items() if any(token in k for token in ("有效发光面", "发光面(mm)", "发光面 ", "光束直径"))),
            None,
        )
        if active_source:
            key, value = active_source
            dims = [float(v) for v in NUMBER.findall(str(value))[:2]]
            if len(dims) == 2:
                spec["active_mm"] = dims
                field_source["active_mm"] = "source_product_parameter:" + key

        # Never infer a physical body from an emitting aperture. The STEP-derived
        # envelope is added only after the CAD geometry audit accepts that asset.
        if "body_mm" not in spec:
            item.setdefault("unknown_fields", [])
            if "verified STEP physical envelope" not in item["unknown_fields"]:
                item["unknown_fields"].append("verified STEP physical envelope")
        counts[category] = counts.get(category, 0) + 1
        changed += 1
    data["revision"] = "2026-10-14.34"
    CATALOG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"revision": data["revision"], "normalized_records": changed, "categories": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
