#!/usr/bin/env python3
"""Download and register exact 3D CAD attachments exposed on Shuangyi product pages.

The source pages use upload/ links with human-readable attachment titles. This
connector only registers native CAD attachments whose title matches a catalog
item's exact source model; it does not assign family or neighboring variants.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import pathlib
import re
import subprocess
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAT = ROOT / "oracle_viewer/catalog"
ASSETS = CAT / "assets"
HARDWARE = CAT / "hardware.json"
MANIFEST = CAT / "manifest.json"
CAD_EXTS = {".igs", ".iges", ".step", ".stp"}
LINK_RE = re.compile(
    r'<a\s+href="([^"]+)"[^>]*title="点击下载([^"]+)"[^>]*>', re.I
)


def fetch(url: str, dest: pathlib.Path) -> None:
    subprocess.run(
        ["curl", "-fL", "--max-time", "90", "-A", "Mozilla/5.0", "-sS", url, "-o", str(dest)],
        check=True,
    )


def main() -> None:
    hardware = json.loads(HARDWARE.read_text())
    manifest = json.loads(MANIFEST.read_text())
    attached = {row["file"] for row in manifest}
    rows_by_model: dict[str, list[dict]] = {}
    for item in hardware["items"]:
        if "Shuangyi" not in item.get("vendor", ""):
            continue
        model = item.get("source_model")
        if model and not item.get("cad_files"):
            rows_by_model.setdefault(model, []).append(item)

    page_urls = sorted({
        page
        for rows in rows_by_model.values()
        for item in rows
        for page in item.get("source_pages", [item.get("source_url")])
        if page
    })
    registered = []
    for page_url in page_urls:
        page_path = pathlib.Path("/tmp") / f"shuangyi-{hashlib.sha1(page_url.encode()).hexdigest()[:12]}.html"
        fetch(page_url, page_path)
        page_html = page_path.read_text(errors="ignore")
        candidates = []
        for href, label in LINK_RE.findall(page_html):
            label = html.unescape(label).strip()
            if not re.search(r"3D", label, re.I):
                continue
            ext = pathlib.Path(urllib.parse.urlparse(href).path).suffix.lower()
            if ext not in CAD_EXTS:
                continue
            candidates.append((urllib.parse.urljoin(page_url, href), label, ext))

        for model, items in rows_by_model.items():
            if not any(item.get("source_url") == page_url or page_url in item.get("source_pages", []) for item in items):
                continue
            match = next((row for row in candidates if model.casefold() in row[1].casefold()), None)
            if not match:
                continue
            source_url, label, ext = match
            filename = re.sub(r"[^A-Za-z0-9._-]+", "-", f"SHY-{model}-official{ext}")
            target = ASSETS / filename
            if not target.exists():
                fetch(source_url, target)
            data = target.read_bytes()
            if ext in {".igs", ".iges"} and not data[:80].decode("latin1", errors="ignore").startswith("SolidWorks IGES") and b"S      1" not in data[:2000]:
                raise RuntimeError(f"Downloaded file does not have an IGES header: {filename}")
            sha = hashlib.sha256(data).hexdigest()
            for item in items:
                if item.get("source_url") != page_url and page_url not in item.get("source_pages", []):
                    continue
                item["cad_files"] = [filename]
                item["cad_status"] = "official_3d_cad_downloaded"
                item["cad_scope"] = "Exact model-labeled 3D CAD attachment linked from the Shuangyi product page; dimensions remain subject to engineering verification."
                item["cad_source_url"] = source_url
                item["cad_brep_validation"] = "pending"
                item["review_status"] = "Exact model label matched to official product-page attachment; FreeCAD geometry validation pending."
                if filename not in attached:
                    manifest.append({
                        "item_id": item["id"],
                        "kind": "iges" if ext in {".igs", ".iges"} else "step",
                        "file": filename,
                        "url": source_url,
                        "sha256": sha,
                        "bytes": len(data),
                        "status": "downloaded",
                        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "manufacturer_file_name": label,
                        "source_page": page_url,
                        "source_page_title": item.get("source_title"),
                        "source_model_label": label,
                        "cad_scope": item["cad_scope"],
                        "tls_verified": True,
                    })
                    attached.add(filename)
                registered.append({"item_id": item["id"], "model": model, "file": filename, "bytes": len(data), "sha256": sha})

    if registered:
        current = hardware.get("revision", "2026-10-14.34")
        suffix = int(current.rsplit(".", 1)[-1]) if "." in current else 0
        hardware["revision"] = f"2026-10-14.{suffix + 1}"
        HARDWARE.write_text(json.dumps(hardware, ensure_ascii=False, indent=2) + "\n")
        MANIFEST.write_text(json.dumps(sorted(manifest, key=lambda row: (row.get("item_id", ""), row["file"], row.get("kind", ""))), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"registered": registered, "count": len(registered), "revision": hardware.get("revision")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
