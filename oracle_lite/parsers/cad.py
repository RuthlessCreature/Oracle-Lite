from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path

from .base import ParsedDocument, ParsedSegment


_STEP_SCHEMA = re.compile(r"FILE_SCHEMA\s*\(\s*\((.*?)\)\s*\)", re.I | re.S)
_STEP_ENTITY = re.compile(r"^\s*#\d+\s*=\s*([A-Z0-9_]+)\s*\(", re.I)
_XT_VERSION = re.compile(r"TRANSMIT FILE created by modeller version\s+([^\s]+)", re.I)
_XT_SCHEMA = re.compile(r"\bSCH_[0-9_]+\b", re.I)
_XT_NODE = re.compile(r"\b(BODY|ASSEMBLY|WORLD|POINTER_LIS_BLOCK|FACE|EDGE|VERTEX|SHELL|REGION)\b", re.I)


def _copy_original(path: Path, asset_dir: Path) -> Path:
    asset_dir.mkdir(parents=True, exist_ok=True)
    target = asset_dir / ("source" + path.suffix.lower())
    shutil.copyfile(path, target)
    return target.resolve()


def parse_step(path: Path, asset_dir: Path) -> ParsedDocument:
    """Bounded streaming metadata parser; preserves the original STEP as truth."""
    original = _copy_original(path, asset_dir)
    schemas: list[str] = []
    entities: Counter[str] = Counter()
    header_lines: list[str] = []
    in_header = False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            upper = line.upper()
            if "HEADER;" in upper:
                in_header = True
            if in_header and len(header_lines) < 128:
                header_lines.append(line.strip())
            if in_header and "ENDSEC;" in upper:
                joined = "\n".join(header_lines)
                match = _STEP_SCHEMA.search(joined)
                if match:
                    schemas = re.findall(r"'([^']+)'", match.group(1))
                in_header = False
            match = _STEP_ENTITY.match(line)
            if match:
                entities[match.group(1).upper()] += 1

    top = entities.most_common(64)
    summary = [
        "Mechanical CAD source (STEP)",
        f"FILE_SCHEMA: {', '.join(schemas) if schemas else 'unknown'}",
        f"Entity count: {sum(entities.values())}",
    ]
    summary.extend(f"{name}: {count}" for name, count in top)
    text = "\n".join(summary)
    return ParsedDocument(
        title=path.name,
        text=text,
        segments=[ParsedSegment(
            text=text,
            images=[],
            metadata={
                "kind": "cad_step_metadata",
                "cad_format": "STEP",
                "schemas": schemas,
                "entity_count": sum(entities.values()),
                "entity_types": dict(top),
                "original_asset": str(original),
                "geometry_label_authoritative": False,
            },
        )],
        metadata={
            "cad_format": "STEP",
            "original_asset": str(original),
            "geometry_preserved": True,
            "geometry_parsed": False,
        },
    )


def parse_parasolid_xt(path: Path, asset_dir: Path) -> ParsedDocument:
    """Stream X_T conservatively; never interprets text tokens as exact geometry."""
    original = _copy_original(path, asset_dir)
    version = None
    schema = None
    nodes: Counter[str] = Counter()
    with path.open("r", encoding="latin-1", errors="replace") as stream:
        for line in stream:
            if version is None:
                match = _XT_VERSION.search(line)
                if match:
                    version = match.group(1)
            if schema is None:
                match = _XT_SCHEMA.search(line)
                if match:
                    schema = match.group(0)
            for node in _XT_NODE.findall(line):
                nodes[node.upper()] += 1

    top = nodes.most_common(32)
    text = "\n".join([
        "Mechanical CAD source (Parasolid XT text)",
        f"Modeller version: {version or 'unknown'}",
        f"Schema: {schema or 'unknown'}",
        *[f"{name}: {count}" for name, count in top],
    ])
    return ParsedDocument(
        title=path.name,
        text=text,
        segments=[ParsedSegment(
            text=text,
            images=[],
            metadata={
                "kind": "cad_parasolid_xt_metadata",
                "cad_format": "PARASOLID_XT_TEXT",
                "modeller_version": version,
                "schema": schema,
                "node_tokens": dict(top),
                "original_asset": str(original),
                "geometry_label_authoritative": False,
            },
        )],
        metadata={
            "cad_format": "PARASOLID_XT_TEXT",
            "original_asset": str(original),
            "geometry_preserved": True,
            "geometry_parsed": False,
        },
    )


def parse_parasolid_xb(path: Path, asset_dir: Path) -> ParsedDocument:
    """Preserve binary XT without pretending to decode proprietary geometry."""
    original = _copy_original(path, asset_dir)
    text = "Mechanical CAD source (Parasolid XT binary); original geometry preserved as an opaque CAD asset."
    return ParsedDocument(
        title=path.name,
        text=text,
        segments=[ParsedSegment(
            text=text,
            images=[],
            metadata={
                "kind": "cad_parasolid_xb_opaque",
                "cad_format": "PARASOLID_XT_BINARY",
                "original_asset": str(original),
                "geometry_label_authoritative": False,
            },
        )],
        metadata={
            "cad_format": "PARASOLID_XT_BINARY",
            "original_asset": str(original),
            "geometry_preserved": True,
            "geometry_parsed": False,
        },
    )
