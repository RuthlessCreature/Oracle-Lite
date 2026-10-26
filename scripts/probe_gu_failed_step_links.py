#!/usr/bin/env python3
"""Recheck advertised GU STEP URLs that did not return a STEP header."""
from __future__ import annotations

import concurrent.futures
import html
import json
import pathlib
import re
import requests

PROGRESS = pathlib.Path("/tmp/gu-optics-step-download-progress.json")
OUT = pathlib.Path(__file__).resolve().parents[1] / "oracle_viewer/catalog/gu_optics_failed_step_probe.json"


def probe(row: tuple[str, dict]) -> dict:
    sku, prior = row
    record = {"model": sku, "url": prior.get("url")}
    try:
        with requests.get(
            record["url"],
            headers={"User-Agent": "Mozilla/5.0 Oracle-Lite catalog verification"},
            stream=True,
            timeout=(10, 30),
        ) as response:
            prefix = next(response.iter_content(1024), b"")
            text = prefix.decode("utf-8", "replace")
            title = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
            record.update(
                http_status=response.status_code,
                content_type=response.headers.get("content-type"),
                content_length_header=response.headers.get("content-length"),
                received_prefix_bytes=len(prefix),
                page_title=html.unescape(re.sub(r"\s+", " ", title.group(1)).strip()) if title else None,
            )
    except Exception as exc:
        record.update(probe_error=repr(exc))
    return record


def main() -> None:
    progress = json.loads(PROGRESS.read_text())
    failures = [(sku, row) for sku, row in progress.items() if row.get("status") != "downloaded"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records = sorted(pool.map(probe, failures), key=lambda row: row["model"])
    OUT.write_text(json.dumps({"probe": "GET advertised STEP URL; inspect HTTP status, MIME, first 1 KiB, and HTML title", "items": records}, ensure_ascii=False, indent=2) + "\n")
    from collections import Counter
    print(json.dumps({"probed": len(records), "http_statuses": dict(Counter(str(row.get("http_status", "error")) for row in records)), "page_titles": dict(Counter(row.get("page_title") or "(empty/no HTML title)" for row in records))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
