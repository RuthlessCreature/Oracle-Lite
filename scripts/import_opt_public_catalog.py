"""Import exact OPT product SKUs listed on the public manufacturer website.

Only model names and parameters published on OPT's own product/performance pages
are imported. Missing fields remain unknown; this does not synthesize variants.
"""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import html
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parents[1] / "oracle_viewer" / "catalog"
SITEMAP = Path("/tmp/opt-sitemap-urls.json")
BASE = "https://www.optmv.com"


def clean(s: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", s)).split())


def get(url: str) -> str:
    r = requests.get(url, timeout=14, headers={"User-Agent": "OracleViewerCatalogResearch/1.0"})
    r.raise_for_status()
    return r.content.decode("utf-8-sig", errors="replace")


def parse_page(url: str):
    m = re.search(r"details(\d+)_(\d+)\.html", url)
    if not m:
        return None
    cat, page_id = m.groups()
    kind = {"17": "light", "75": "lens", "88": "camera"}.get(cat)
    if not kind:
        return None
    page = get(url)
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.I | re.S)
    model = clean(h1.group(1)) if h1 else ""
    # The product selector pages are not individual SKUs.
    if not model or model == "产品":
        return None
    perf_match = re.search(r"data-url=[\"']([^\"']*performance_details" + cat + r"_" + page_id + r"\.html)[\"']", page, re.I)
    if not perf_match:
        return None
    perf_url = urljoin(BASE, perf_match.group(1))
    perf = get(perf_url)
    params = {}
    for a, b in re.findall(r"<tr[^>]*>\s*<th[^>]*>(.*?)</th>\s*<td[^>]*>(.*?)</td>", perf, re.I | re.S):
        key, value = clean(a), clean(b)
        if key and value:
            params[key] = value
    model = params.get("型号", model)
    if not re.search(r"[A-Za-z0-9]", model) or len(model) > 100:
        return None
    spec = {"manufacturer_parameters": params}
    if kind == "camera":
        resolution = re.search(r"([\d,]+)\s*[×xX]\s*([\d,]+)", params.get("分辨率", ""))
        if resolution:
            width, height = (int(x.replace(",", "")) for x in resolution.groups())
            spec.update(width_px=width, height_px=height)
            spec["megapixels"] = round(width * height / 1_000_000, 3)
            if width * height >= 25_000_000:
                spec["catalog_priority"] = "common_aoi_baseline_25mp_plus"
        pixel = re.search(r"([\d.]+)\s*[×xX]\s*([\d.]+)", params.get("像元尺寸(μ㎡)", ""))
        if pixel:
            spec["pixel_um"] = [float(pixel.group(1)), float(pixel.group(2))]
        sensor = params.get("芯片尺寸")
        if sensor: spec["sensor_format"] = sensor
        if params.get("光学接口"): spec["mount"] = params["光学接口"]
        if params.get("快门类型"): spec["shutter"] = params["快门类型"]
        if params.get("数据接口"): spec["interface"] = params["数据接口"]
    elif kind == "lens":
        mappings = {"焦距(mm)": "focal_length_mm", "工作距离(mm)": "working_distance_mm", "放大倍率": "magnification", "靶面Φ(mm)": "image_circle_mm", "景深(mm)": "dof_mm", "畸变": "distortion", "重量（kg）": "weight_kg"}
        for source, dest in mappings.items():
            if source in params:
                nums = re.findall(r"[-+]?\d+(?:\.\d+)?", params[source].replace(",", ""))
                if nums:
                    vals = [float(x) for x in nums]
                    spec[dest] = vals[0] if len(vals) == 1 else vals
        if params.get("镜头接口"): spec["mount"] = params["镜头接口"]
        if params.get("镜头类型"): spec["lens_type"] = params["镜头类型"]
    else:
        for source, dest in {"发光面尺寸（长）(mm)": "emitting_length_mm", "发光面尺寸（宽）(mm)": "emitting_width_mm", "消耗功率": "power_description", "颜色": "color", "建议配套控制器": "recommended_controller"}.items():
            if params.get(source): spec[dest] = params[source]
        name = model.upper()
        spec["type"] = "backlight" if "FL" in name else "coaxial" if "CO" in name or "LSSC" in name else "dome" if "RID" in name else "bar" if "LI" in name else "ring"
    known = {"manufacturer_parameters", "width_px", "height_px", "megapixels", "catalog_priority", "pixel_um", "sensor_format", "mount", "shutter", "interface", "focal_length_mm", "working_distance_mm", "magnification", "image_circle_mm", "dof_mm", "distortion", "weight_kg", "lens_type", "emitting_length_mm", "emitting_width_mm", "power_description", "color", "recommended_controller", "type"}
    item = {
        "id": model, "kind": kind, "vendor": "OPT", "source_url": perf_url,
        "product_url": url, "datasheet_url": None, "spec": spec, "price": None,
        "cad_status": "not_downloaded", "field_source": {k: "manufacturer_parameter_page" for k in spec},
        "review_status": "manufacturer_parameter_page", "unknown_fields": ["price", "availability", "vendor_CAD", "housing_geometry"],
        "layout_adapter_ready": False,
    }
    if kind == "camera":
        item["unknown_fields"] += ["body_dimensions", "connector_clearance", "lens_flange_geometry", "full_sensor_roi_verification"]
    if kind == "lens":
        item["unknown_fields"] += ["body_dimensions", "distortion_test_conditions", "mount_flange_geometry"]
    record = {"item": item, "url": url, "perf_url": perf_url, "source_model": model}
    return record


def main():
    urls = json.loads(SITEMAP.read_text())
    # Cameras first (with 25 MP+ priority), then lenses. This batch is focused
    # on large single-shot fields of view; lights can follow in a separate pass.
    candidates = [u for u in urls if re.search(r"/content/details(?:75|88)_\d+\.html$", u)]
    catalog_path = ROOT / "hardware.json"
    catalog = json.loads(catalog_path.read_text())
    existing_ids = {x["id"].casefold() for x in catalog["items"]}
    found, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        futures = {pool.submit(parse_page, u): u for u in candidates}
        for ix, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                row = future.result()
                if row:
                    found.append(row)
            except Exception as exc:
                errors.append({"url": futures[future], "error": str(exc)[:250]})
            if ix % 100 == 0:
                print(f"OPT source pages checked: {ix}/{len(futures)}; exact SKU pages parsed: {len(found)}", flush=True)
    # Prefer larger sensors first among cameras, then retain source page order by SKU.
    unique = {}
    for row in found:
        item = row["item"]
        unique.setdefault(item["id"].casefold(), row)
    new = [r for k, r in unique.items() if k not in existing_ids]
    new.sort(key=lambda r: (0 if r["item"]["kind"] == "camera" and r["item"]["spec"].get("megapixels", 0) >= 25 else 1, {"camera": 0, "lens": 1, "light": 2}[r["item"]["kind"]], r["item"]["id"]))
    master_path = ROOT / "batches" / "hardware-expansion-2000.json"
    master = json.loads(master_path.read_text())
    remaining = max(0, 2000 - master.get("added_count", 0))
    chosen = new[:remaining]
    catalog["items"].extend(r["item"] for r in chosen)
    old_rev = catalog.get("revision", "2026-10-21.7")
    rev_match = re.match(r"(.*\.)(\d+)$", old_rev)
    catalog["revision"] = f"{rev_match.group(1)}{int(rev_match.group(2))+1}" if rev_match else old_rev + ".1"
    catalog["notes"].append("OPT public product SKUs imported from the official OPT website; manufacturer-reported parameters retained verbatim; absent CAD/housing geometry is not fabricated.")
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
    added_ids = [r["item"]["id"] for r in chosen]
    master["added_ids"] = list(dict.fromkeys(master.get("added_ids", []) + added_ids))
    master["added_count"] = len(master["added_ids"])
    master["remaining_to_target"] = max(0, 2000 - master["added_count"])
    master.setdefault("sources", []).append({"name": "OPT official product / performance pages", "added_count": len(added_ids), "source_urls": list(dict.fromkeys(u for r in chosen for u in [r["url"], r["perf_url"]]))})
    master_path.write_text(json.dumps(master, ensure_ascii=False, indent=2))
    batch = {"batch": "opt-public-skus-2026-10-21", "source": "https://www.optmv.com/", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "candidate_urls": len(candidates), "parsed_exact_skus": len(unique), "new_exact_skus_found": len(new), "added_count": len(added_ids), "added_ids": added_ids, "failures_count": len(errors), "failures_sample": errors[:30]}
    (ROOT / "batches" / "opt-public-skus-2026-10-21.json").write_text(json.dumps(batch, ensure_ascii=False, indent=2))
    print(json.dumps({"candidate_pages": len(candidates), "exact_skus": len(unique), "new_found": len(new), "added": len(added_ids), "remaining": master["remaining_to_target"], "kinds_added": {k: sum(r["item"]["kind"] == k for r in chosen) for k in ["camera", "lens", "light"]}, "25mp_plus_cameras_added": sum(r["item"]["kind"] == "camera" and r["item"]["spec"].get("megapixels", 0) >= 25 for r in chosen), "failures": len(errors)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
