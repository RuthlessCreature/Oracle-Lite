#!/usr/bin/env python3
"""Fetch exact-model HIKROBOT 3D profiler CAD packages from the public distributor library.

The original ZIPs are retained. Only STEP/STP/IGES/IGS/DXF source files are
extracted into the local asset library; large display-only OBJ files remain in
the archived package. Model labels are taken from the CAD member names, and
variant suffixes/revisions are kept as separate catalog items.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import unicodedata
import urllib.parse
import zipfile
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "oracle_viewer/catalog"
ASSETS = CAT / "assets"
ARCHIVES = CAT / "source_archives/hikrobot/mvisionpro"
DOCS = "https://mvisionpro.com/hikrobot/documents/"
CAD_EXTS = {".step", ".stp", ".iges", ".igs", ".dxf"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                            prefix=path.name + ".", suffix=".tmp", delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        temp = Path(f.name)
    os.replace(temp, path)


def safe_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return value or "unnamed"


def model_code(stem: str, fallback: str) -> str:
    # HIKROBOT archives sometimes append Chinese drawing-description text to
    # filenames without setting ZIP's UTF-8 flag. Retain the exact SKU/revision
    # prefix and drop only trailing descriptive characters.
    m = re.match(r"^(MV[-_][A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*(?:\s+V\d+(?:\.\d+)*)?)", stem.strip(), re.I)
    return m.group(1).replace("_", "-") if m else fallback.strip()


def read_zip_name(blob: bytes) -> str:
    # HIKROBOT packages are ZIPs and normally mark UTF-8 names. Legacy
    # non-UTF8 entries are decoded as GBK before falling back to CP437.
    import io
    for encoding in (None, "gbk", "cp437"):
        try:
            with zipfile.ZipFile(io.BytesIO(blob), metadata_encoding=encoding) as zf:
                if encoding is not None:
                    return encoding
                return "utf8"
        except (UnicodeDecodeError, RuntimeError):
            continue
    return "cp437"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Oracle-AOI-CAD-catalog/1.0"})
    response = session.get(DOCS, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    section = next((s for s in soup.select("section.cat-section")
                    if "3D cameras & profilers" in s.get_text(" ", strip=True)[:100]), None)
    if section is None:
        raise RuntimeError("HIKROBOT profiler CAD section not found; refusing an empty ingest")

    hardware_path, manifest_path = CAT / "hardware.json", CAT / "manifest.json"
    hardware, manifest = json.loads(hardware_path.read_text()), json.loads(manifest_path.read_text())
    items = {i["id"]: i for i in hardware["items"]}
    linked = {(a.get("item_id"), a.get("file")) for a in manifest}
    time_now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds")
    registered, packages, errors = [], [], []

    for article in section.select("article"):
        title_link = article.select_one("h2 a[href], h3 a[href], h4 a[href]")
        product_url = urllib.parse.urljoin(DOCS, title_link["href"]) if title_link else DOCS
        for link in article.select('a[data-document-type="drawing"][data-document-action="download"]'):
            archive_name = link.get("download", "")
            package_url = urllib.parse.urljoin(DOCS, link.get("href", ""))
            if not archive_name.lower().endswith(".zip") or not package_url:
                continue
            archive_model = Path(archive_name).stem
            archive_path = ARCHIVES / f"{safe_name(archive_model)}.zip"
            try:
                if archive_path.exists():
                    blob = archive_path.read_bytes()
                else:
                    r = session.get(package_url, timeout=120)
                    r.raise_for_status()
                    blob = r.content
                    if not blob.startswith(b"PK\x03\x04"):
                        raise ValueError("download response is not a ZIP archive")
                    archive_path.write_bytes(blob)
                archive_sha = digest(blob)
                with zipfile.ZipFile(archive_path) as zf:
                    members = [x for x in zf.infolist()
                               if not x.is_dir() and PurePosixPath(x.filename).suffix.lower() in CAD_EXTS]
                    for member in members:
                        # Block path traversal; exports are always renamed into assets/.
                        member_path = PurePosixPath(member.filename)
                        if member_path.is_absolute() or ".." in member_path.parts:
                            errors.append({"model": archive_model, "member": member.filename,
                                           "error": "unsafe archive path skipped"})
                            continue
                        original = zf.read(member)
                        ext = member_path.suffix.lower()
                        raw_label = member_path.stem.strip()
                        model_label = model_code(raw_label, link.get("data-document-model", archive_model))
                        if not model_label or not re.search(r"MV[-_ ]", model_label, re.I):
                            # Product-page SKU is exact, but an unlabeled artifact
                            # is not safe to attach to an item without manual review.
                            errors.append({"model": archive_model, "member": member.filename,
                                           "error": "CAD member has no recognizable HIKROBOT model label"})
                            continue
                        item_id = model_label
                        filename = f"HIK-{safe_name(model_label)}-{digest(original)[:10]}{ext}"
                        target = ASSETS / filename
                        if not target.exists():
                            target.write_bytes(original)
                        item = items.get(item_id)
                        if item is None:
                            item = {
                                "id": item_id, "kind": "surface_sensor", "vendor": "HIKROBOT",
                                "source_model": model_label, "source_url": product_url,
                                "product_url": product_url, "datasheet_url": None,
                                "spec": {"manufacturer_model": model_label,
                                         "sensor_class": "3D camera / profiler; verify exact family variant"},
                                "price": None, "cad_status": "exact_model_label_distributor_CAD_downloaded_geometry_pending",
                                "cad_files": [], "cad_model_ref": model_label,
                                "cad_scope": "Exact model label is present in a public HIKROBOT distributor drawing package. The original archive is retained; per-file topology and dimensional correspondence are recorded after geometry audit.",
                                "review_status": "CAD package downloaded; per-format B-Rep and published dimensions require audit",
                                "unknown_fields": ["measurement specifications", "mounting datum", "outer-envelope and cable clearance", "availability and revision status"],
                                "layout_adapter_ready": False, "cad_brep_validation": "pending",
                                "field_source": {"manufacturer_model": product_url}
                            }
                            items[item_id] = item
                        if filename not in item.setdefault("cad_files", []):
                            item["cad_files"].append(filename)
                        if (item_id, filename) not in linked:
                            row = {
                                "item_id": item_id, "kind": ext.lstrip("."), "file": filename,
                                "url": package_url, "source_page": product_url,
                                "document_library_url": DOCS, "direct_package_url": package_url,
                                "sha256": digest(original), "bytes": len(original), "status": "downloaded",
                                "retrieved_at": time_now, "tls_verified": True,
                                "source_archive_file": str(archive_path.relative_to(CAT)),
                                "source_archive_sha256": archive_sha, "source_archive_bytes": len(blob),
                                "source_archive_member": member.filename,
                                "manufacturer_file_name": member_path.name,
                                "raw_model_label_from_member": raw_label,
                                "source_model_label": model_label,
                                "cad_scope": "Exact product-model label copied from distributor's original ZIP member; no neighboring SKU substitution."
                            }
                            manifest.append(row)
                            linked.add((item_id, filename))
                        registered.append({"item_id": item_id, "kind": ext.lstrip("."),
                                           "file": filename, "source_member": member.filename,
                                           "archive": str(archive_path.relative_to(CAT))})
                packages.append({"model": archive_model, "archive": str(archive_path.relative_to(CAT)),
                                 "bytes": len(blob), "sha256": archive_sha,
                                 "cad_members": len(members)})
            except Exception as exc:  # Preserve batch progress; report individual source failures.
                errors.append({"model": archive_model, "url": package_url,
                               "error": f"{type(exc).__name__}: {exc}"})

    if registered:
        current = hardware.get("revision", "2026-10-14.42")
        suffix = int(current.rsplit(".", 1)[-1]) if "." in current else 0
        hardware["revision"] = f"2026-10-14.{suffix + 1}"
        hardware["items"] = sorted(items.values(), key=lambda x: (x.get("kind", ""), x.get("vendor", ""), x["id"]))
        manifest.sort(key=lambda a: (a.get("item_id", ""), a["file"], a.get("kind", "")))
        atomic_json(hardware_path, hardware)
        atomic_json(manifest_path, manifest)
    print(json.dumps({"catalog_revision": hardware.get("revision"),
                      "packages": packages, "registered": registered, "errors": errors},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
