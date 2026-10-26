"""Offline source-edge audit. Construction success is not metrology acceptance.

Run with the isolated CAD interpreter, never through the TALKER model process.
Reports retain source node identifiers and failures rather than dropping edges.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


def audit_edges(brep, *, source_unit, source_sha256, progress=None, max_edges=None):
    from parasolid_kit.interop.occt.geometry import GeometryFactory
    from parasolid_kit.interop.occt.options import OcctConversionOptions
    from parasolid_kit.interop.occt.parametric import ParametricTopologyBuilder

    if not brep.complete or not brep.topology.valid:
        raise ValueError(
            "Complete source decoding and valid source topology are required"
        )
    options = OcctConversionOptions(source_unit=source_unit, target_unit="mm")
    builder = ParametricTopologyBuilder(brep, GeometryFactory(options))
    builder._build_vertices()
    started = time.monotonic()
    failures = []
    checked = 0

    def report():
        return {
            "audit": "source_edge_construction",
            "source_sha256": source_sha256,
            "options": options.to_dict(),
            "source_complete": True,
            "source_topology_valid": True,
            "source_edge_count": len(brep.edges),
            "checked_edge_count": checked,
            "constructed_edge_count": len(builder.edge_shapes),
            "all_edges_checked": checked == len(brep.edges),
            "failed_edge_count": len(failures),
            "failure_categories": dict(Counter(item["error"] for item in failures)),
            "failures": failures,
            "construction_operations": list(dict.fromkeys(builder.operations)),
            "geometry_checks": dict(builder.geometry_checks),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "metrology_ready": False,
            "qualification": "Edge construction only; face trims, shells, solids and coordinate registration require separate validation.",
        }

    for edge in brep.edges:
        if max_edges is not None and checked >= max_edges:
            break
        try:
            builder._build_edge(edge)
        except Exception as error:
            curve = builder.curves.get(edge.curve)
            try:
                effective_tolerance = builder._edge_tolerance(edge)
            except ValueError:
                effective_tolerance = None
            failures.append(
                {
                    "edge_id": edge.id,
                    "source_node_index": edge.source.node_index,
                    "source_node_id": edge.source.node_id,
                    "curve_id": edge.curve,
                    "curve_kind": curve.kind.value if curve is not None else None,
                    "source_tolerance": edge.tolerance,
                    "effective_edge_tolerance_mm": effective_tolerance,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        checked += 1
        if progress is not None and (checked % 100 == 0 or checked == len(brep.edges)):
            progress(report())
    result = report()
    if progress is not None:
        progress(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source-unit", required=True, choices=("m", "cm", "mm", "in", "ft")
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "third_party"
        / "parasolid-schemas",
    )
    parser.add_argument("--max-edges", type=int)
    args = parser.parse_args()
    if args.max_edges is not None and args.max_edges < 1:
        parser.error("--max-edges must be positive")
    from parasolid_kit import read_brep

    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    decoded = read_brep(args.source, schema_dir=str(args.schema_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(report):
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(args.output)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "checked_edge_count",
                        "constructed_edge_count",
                        "failed_edge_count",
                        "elapsed_seconds",
                    )
                }
            ),
            flush=True,
        )

    result = audit_edges(
        decoded.brep,
        source_unit=args.source_unit,
        source_sha256=digest,
        progress=save,
        max_edges=args.max_edges,
    )
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Source file changed during audit")
    return 0 if result["all_edges_checked"] and not result["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
