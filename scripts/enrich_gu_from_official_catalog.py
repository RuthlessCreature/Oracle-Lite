#!/usr/bin/env python3
"""Enrich GU Optics listings from the locally archived official catalog 9.0 PDF."""
from __future__ import annotations

import collections
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG = ROOT / "oracle_viewer/catalog/hardware.json"
PDF = ROOT / "oracle_viewer/catalog/source_documents/gu-optics/gu-optics-electronic-catalog-9.0.pdf"
PDF_REL = PDF.relative_to(ROOT).as_posix()
FAMILY_RE = re.compile(r"(?<!\d)(66\d{4})(?:-([RGBW]))?(?!\d)")
VALUE_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:\s*[*×xX]\s*\d+(?:\.\d+)?){0,2}(?:\s*-\s*\d+(?:\.\d+)?)?(?![\w.])")
COLOR_MAP = {"R": "Red", "G": "Green", "B": "Blue", "W": "White"}


def dim(token: str) -> list[float] | None:
    values = re.findall(r"\d+(?:\.\d+)?", token)
    return [float(x) for x in values] if len(values) in (2, 3) else None


def numeric(token: str) -> float | None:
    if "*" in token or "×" in token or "x" in token.lower() or "-" in token[1:]:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def rows_from_pdf() -> dict[str, dict]:
    text = subprocess.run(["pdftotext", "-layout", str(PDF), "-"], check=True, stdout=subprocess.PIPE).stdout.decode("utf-8", "replace")
    pages = text.split("\f")
    rows: dict[str, dict] = {}
    for page_number in range(172, min(177, len(pages) + 1)):
        for line in pages[page_number - 1].splitlines():
            matches = list(FAMILY_RE.finditer(line))
            families = list(dict.fromkeys(match.group(1) for match in matches))
            for family in families:
                family_matches = [match for match in matches if match.group(1) == family]
                tail_start = family_matches[-1].end()
                tail_end = next((match.start() for match in matches if match.start() > family_matches[-1].start() and match.group(1) != family), len(line))
                tail = line[tail_start:tail_end].strip()
                tokens = [re.sub(r"\s+", "", token) for token in VALUE_RE.findall(tail)]
                if len(tokens) >= 2 and (family not in rows or len(tokens) > len(rows[family]["tokens"])):
                    rows[family] = {"pdf_page": page_number, "tail": tail, "tokens": tokens}
    return rows


def controller_rows_from_pdf() -> dict[str, dict]:
    text = subprocess.run(["pdftotext", "-layout", str(PDF), "-"], check=True, stdout=subprocess.PIPE).stdout.decode("utf-8", "replace")
    pages = text.split("\f")
    rows: dict[str, dict] = {}
    for page_number in range(172, min(177, len(pages) + 1)):
        for line in pages[page_number - 1].splitlines():
            match = re.match(r"\s*(6614\d{2})\s+(.+)$", line)
            if match:
                rows[match.group(1)] = {"pdf_page": page_number, "row": re.sub(r"\s+", " ", match.group(2)).strip()}
    return rows


def main() -> None:
    family_rows = rows_from_pdf()
    controller_rows = controller_rows_from_pdf()
    data = json.loads(CATALOG.read_text())
    counts: collections.Counter[str] = collections.Counter()
    conflicts: list[dict] = []
    missing_family: list[str] = []
    for item in data["items"]:
        if not item.get("vendor", "").startswith("GU Optics"):
            continue
        model = str(item.get("source_model", ""))
        family_match = re.match(r"(66\d{4})", model)
        if not family_match:
            continue
        family = family_match.group(1)
        row = family_rows.get(family)
        if not row:
            missing_family.append(model)
            continue
        category = item.get("source_category")
        tokens = row["tokens"]
        spec = item.setdefault("spec", {})
        field_source = item.setdefault("field_source", {})
        catalog_source = f"{PDF_REL}#pdf-page-{row['pdf_page']}"
        spec["manufacturer_catalog_version"] = "GU Optics Electronic Catalog 9.0"
        spec["manufacturer_catalog_pdf_page"] = row["pdf_page"]
        spec["manufacturer_catalog_row"] = row["tail"]
        field_source["manufacturer_catalog_version"] = catalog_source
        field_source["manufacturer_catalog_pdf_page"] = catalog_source
        field_source["manufacturer_catalog_row"] = catalog_source
        item["manufacturer_catalog_document"] = PDF_REL

        layout: dict[str, int] | None = None
        if category == "ring":
            layout = {"led_rows": 0, "voltage_v": 1, "power_group": 2, "power_red": 3, "working_distance": 4}
        elif category in ("bar", "uniform_bar"):
            layout = {"led_rows": 0, "active": 1, "voltage_v": 2, "power_group": 3, "power_red": 4}
        elif category == "combined_bar":
            layout = {"led_rows": 0, "voltage_v": 1, "power_group": 2, "power_red": 3}
        elif category in ("backlight", "parallel_backlight", "side_backlight"):
            layout = {"active": 0, "voltage_v": 1, "power_group": 2, "power_red": 3, "envelope": 4}
        elif category == "hole_backlight":
            layout = {"active": 0, "aperture": 1, "voltage_v": 2, "power_group": 3, "power_red": 4, "envelope": 5}
        elif category == "coaxial":
            layout = {"active": 0, "voltage_v": 1, "power_group": 2, "power_red": 3}
        elif category == "dome":
            layout = {"emitting_diameter": 0, "aperture": 1, "voltage_v": 2, "power_group": 3, "power_red": 4}
        elif category == "spot":
            layout = {"interface_diameter": 0, "emitting_diameter": 1, "voltage_v": 2, "power_group": 3, "power_red": 4, "envelope": 5}
        elif category in ("ringless", "squareless"):
            layout = {"manufacturer_dimensions": 0, "voltage_v": 1, "power_group": 2, "power_red": 3}
        elif category == "high_intensity_line":
            layout = {"active": 0, "working_distance": 1, "voltage_v": 2, "power_group": 3, "power_red": 4, "manufacturer_dimensions": 5}
        if not layout:
            continue
        if len(tokens) <= max(layout.values()):
            counts[category + ":short_row"] += 1
            continue

        for field, index in layout.items():
            token = tokens[index]
            source = catalog_source
            if field == "led_rows":
                if (value := numeric(token)) is not None:
                    spec.setdefault("led_rows", int(value)); field_source.setdefault("led_rows", source)
            elif field == "voltage_v":
                if (value := numeric(token)) is not None:
                    if spec.get("voltage_v") is not None and abs(float(spec["voltage_v"]) - value) > 0.01:
                        conflicts.append({"model": model, "field": field, "product_page": spec["voltage_v"], "catalog": value})
                    else:
                        spec["voltage_v"] = value; field_source["voltage_v"] = source
            elif field in ("power_group", "power_red"):
                color_code = model[-1:] if model[-1:] in COLOR_MAP else None
                if field == "power_group" and color_code == "R":
                    continue
                if field == "power_red" and color_code != "R":
                    continue
                value = numeric(token)
                if value is None:
                    continue
                catalog_power = value
                spec["catalog_power_w"] = catalog_power
                field_source["catalog_power_w"] = source + ("; RGBW column" if color_code != "R" else "; red column")
                if spec.get("power_w") is not None and abs(float(spec["power_w"]) - catalog_power) > 0.05:
                    conflicts.append({"model": model, "field": "power_w", "product_page": spec["power_w"], "catalog": catalog_power})
                else:
                    spec["power_w"] = catalog_power
                    field_source["power_w"] = source + ("; RGBW column" if color_code != "R" else "; red column")
            elif field == "active":
                dimensions = dim(token)
                if dimensions:
                    spec.setdefault("active_mm", dimensions[:2]); field_source.setdefault("active_mm", source)
            elif field in ("aperture", "interface_diameter", "emitting_diameter"):
                value = numeric(token)
                if value is not None:
                    name = {"aperture": "aperture_diameter_mm", "interface_diameter": "interface_diameter_mm", "emitting_diameter": "emitting_diameter_mm"}[field]
                    spec[name] = value; field_source[name] = source
                    if field == "emitting_diameter":
                        spec.setdefault("active_mm", [value, value]); field_source.setdefault("active_mm", source)
            elif field == "working_distance":
                match = re.fullmatch(r"(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)", token)
                if match:
                    values = [float(match.group(1)), float(match.group(2))]
                    spec["working_distance_range_mm"] = values; field_source["working_distance_range_mm"] = source
            elif field in ("envelope", "manufacturer_dimensions"):
                dimensions = dim(token)
                if dimensions:
                    spec["manufacturer_outer_dimensions_mm"] = dimensions
                    field_source["manufacturer_outer_dimensions_mm"] = source
        if spec.get("power_w") and spec.get("voltage_v"):
            spec["current_a"] = round(float(spec["power_w"]) / float(spec["voltage_v"]), 6)
            field_source["current_a"] = "derived_from_source_power_w_divided_by_voltage_v"
        counts[category] += 1

    conflicts_by_model: dict[str, list[dict]] = collections.defaultdict(list)
    for conflict in conflicts:
        conflicts_by_model[conflict["model"]].append(conflict)
    for item in data["items"]:
        model = item.get("source_model")
        if model in conflicts_by_model:
            item["source_conflicts"] = conflicts_by_model[model]
            note = "官网商品页与电子目录 9.0 的功率字段有差异，采购/控制器选型前需复核当前规格书。"
            if note not in item.get("review_status", ""):
                item["review_status"] = (item.get("review_status", "") + " " + note).strip()

    data["revision"] = "2026-10-14.34"
    CATALOG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    report = {"revision": data["revision"], "matched_family_rows": len(family_rows), "controller_rows": controller_rows, "enriched_records_by_category": dict(counts), "unmatched_model_count": len(missing_family), "unmatched_models": sorted(set(missing_family)), "source_conflict_count": len(conflicts), "source_conflicts": conflicts}
    (ROOT / "oracle_viewer/catalog/gu_official_catalog_enrichment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_conflicts" and k != "unmatched_models"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
