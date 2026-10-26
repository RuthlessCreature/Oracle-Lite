"""Retrieve MSOP point/feature evidence on a CAD mesh for local-model review.

The code is drawing-agnostic. Coordinate-frame hypotheses are supplied as a
project calibration artifact and are scored against the CAD mesh at run time.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oracle_viewer.candidate_mining import (
    COORDINATE_POINT, extract_feature_labels, extract_named_points,
    project_points_to_common_face, referenced_points, screen_point_references,
)
from oracle_viewer.store import ViewerStore


def _systems(text):
    return set(re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", text.upper()))


def _candidate_point_sets(item, detail, pages, named_points):
    from oracle_viewer.msop_coordinates import _coordinate_rows
    labels = extract_feature_labels(detail['methods'], item.get('features', []))
    rows = _coordinate_rows(detail['methods'], [], labels, pages, named_points)
    return rows, 'program-and-alignment-scoped coordinate definitions and references'


def _frame_hypotheses(item, point_rows, detail, calibration):
    method_systems = set()
    for page in detail["methods"]:
        method_systems |= _systems(page.get("measured_system") or "")
        method_systems |= _systems(page.get("text", "")[:1200])
    source_systems = set()
    for row in point_rows:
        for page_no in row.get("pages", []):
            page = next((p for p in calibration["method_pages"] if p.get("page") == page_no), None)
            if page:
                source_systems |= _systems(page.get("measured_system") or "")
                source_systems |= _systems(page.get("text", "")[:1200])
    relevant = method_systems | source_systems
    from oracle_viewer.candidate_mining import extract_named_points
    frame_programs={}
    for value in extract_named_points(calibration['method_pages']).values():
        for occurrence in value.get('occurrences',[]):
            frame_programs.setdefault(occurrence.get('coordinate_system'),set()).add(occurrence.get('program'))
    hypotheses = []
    for hypothesis in calibration["hypotheses"]:
        applicable = set(hypothesis.get("systems", []))
        if not applicable or not relevant or applicable & relevant:
            if all((not hframe or not applicable or hframe in applicable) and
                   ((hypothesis.get('program') is not None and hypothesis.get('program')==r.get('program')) or
                    (not hypothesis.get('program') and len(frame_programs.get(hframe,set()))<=1))
                   for r in point_rows for hframe in [r.get('coordinate_system')]):
                hypotheses.append(hypothesis)
    return hypotheses


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--calibration", type=Path, required=True,
                    help="JSON coordinate-frame calibration hypotheses for this CAD/MSOP pair")
    ap.add_argument("--origin", default="http://127.0.0.1:7870")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fai", type=int, action="append")
    ap.add_argument("--recheck-pending", action="store_true",
                    help="re-evaluate existing unreviewed geometry candidates; dry-run is read-only")
    ap.add_argument("--recheck-model-pending", action="store_true",
                    help="refine unreviewed local-model anchors while preserving its selected-face hypothesis")
    args = ap.parse_args()

    store = ViewerStore(ROOT / "talker/state/viewer")
    project = store.get(args.project)
    mesh = store.mesh(args.project)
    calibration_data = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration_data.get('cad_sha256')!=mesh['source_sha256'] or calibration_data.get('msop_sha256')!=store.active_msop(project)['sha256']:
        raise ValueError('Calibration does not match this CAD/MSOP source pair')
    pages = store.active_msop(project)["method_pages"]
    # The parsed page inventory is read-only evidence for associating point rows
    # with the coordinate-system label that defined them.
    calibration = {**calibration_data, "method_pages": pages}
    named_points = extract_named_points(pages)

    vertices_list, face_ids_list, triangles_list, triangle_face_ids = [], [], [], []
    for face in mesh["faces"]:
        coords = np.asarray(face["positions"], dtype=np.float32).reshape(-1, 3)
        vertices_list.append(coords)
        face_ids_list.extend([face["id"]] * len(coords))
        indices = np.asarray(face["indices"], dtype=np.int64).reshape(-1, 3)
        triangles_list.append(coords[indices])
        triangle_face_ids.extend([face["id"]] * len(indices))
    vertices = np.concatenate(vertices_list)
    face_ids = np.asarray(face_ids_list, dtype=object)
    triangles = np.concatenate(triangles_list)
    triangle_face_ids = np.asarray(triangle_face_ids, dtype=object)

    output = []
    for item in store.items(project):
        if args.fai and item["number"] not in args.fai:
            continue
        old = item.get("annotation")
        allowed_origins = set()
        if args.recheck_pending:
            allowed_origins.add("geometry_candidate")
        if args.recheck_model_pending:
            allowed_origins.add("local_model_candidate")
        if item.get("binding") or (old and not (
                old.get("origin") in allowed_origins and old.get("status") == "pending_review")):
            continue
        detail = store.detail(args.project, item["number"])
        point_rows, retrieval = _candidate_point_sets(item, detail, pages, named_points)
        if not point_rows:
            output.append({"fai": item["number"], "result": "no_explicit_cross_page_geometry_reference"})
            continue
        if any(r.get('coordinate_conflict') for r in point_rows):
            output.append({'fai':item['number'],'result':'conflicting_source_coordinates','point_names':[r['name'] for r in point_rows if r.get('coordinate_conflict')]})
            continue
        hypotheses = _frame_hypotheses(item, point_rows, detail, calibration)
        # During a pending-candidate recheck, keep the model/retrieval's face
        # hypothesis fixed and refine only the XYZ point on those faces. This
        # prevents XY-overlapping assembly layers from silently swapping faces.
        projection_triangles, projection_face_ids = triangles, triangle_face_ids
        candidate_face_ids = ((old or {}).get("candidate_face_ids")
                              or (old or {}).get("face_ids", []))
        if (args.recheck_pending or args.recheck_model_pending) and candidate_face_ids:
            allowed = np.isin(triangle_face_ids, np.asarray(candidate_face_ids, dtype=object))
            if np.any(allowed):
                projection_triangles = triangles[allowed]
                projection_face_ids = triangle_face_ids[allowed]
        scored = []
        for hypothesis in hypotheses:
            result = screen_point_references(
                point_rows, vertices, face_ids, hypothesis,
                max_xy_residual_mm=float(calibration_data.get("max_xy_residual_mm", 1.0)),
                min_inlier_ratio=float(calibration_data.get("min_inlier_ratio", 0.5)),
                min_inliers=int(calibration_data.get("min_inliers", 2)),
                triangles=projection_triangles, triangle_face_ids=projection_face_ids)
            scored.append((result["valid"], result["inlier_ratio"],
                           -float(np.median([r["residual_mm"] for r in result["projected"]])),
                           hypothesis, result))
        if not scored:
            output.append({"fai": item["number"], "result": "no_applicable_frame_calibration",
                           "retrieval": retrieval, "points": len(point_rows)})
            continue
        _, _, _, transform, result = max(scored, key=lambda row: row[:3])
        if not result["valid"]:
            output.append({"fai": item["number"], "result": "weak_geometric_support",
                           "retrieval": retrieval, "points": result["point_count"],
                           "inliers": result["inlier_count"], "inlier_ratio": round(result["inlier_ratio"], 3)})
            continue

        # Explicit flatness constructions are one feature group. A same-page
        # location set whose source Z is constant is also a shared-level group;
        # its points must not be independently assigned to different CAD
        # layers. Joint selection avoids XY-overlapping assembly faces splitting
        # either group.
        method_text = "\n".join(page.get("text", "") for page in detail["methods"])
        common_face = None
        flatness_group = (re.search(r"construct\s+(?:a\s+)?plane\b", method_text, re.I)
                          and re.search(r"dimension\s+flatness\b", method_text, re.I))
        source_z = [float(row["xyz"][2]) for row in point_rows if len(row.get("xyz", [])) >= 3]
        same_source_z_group = (len(point_rows) >= 3 and len(source_z) == len(point_rows)
                               and max(source_z) - min(source_z) <= float(
                                   calibration_data.get("same_source_z_tolerance_mm", 0.001)))
        common_group_reason = ("constructed_plane_flatness" if flatness_group else
                               "constant_source_z" if same_source_z_group else None)
        if flatness_group and candidate_face_ids:
            common_face = project_points_to_common_face(
                result["inliers"], projection_triangles, projection_face_ids,
                candidate_face_ids,
                max_xy_residual_mm=float(calibration_data.get("max_xy_residual_mm", 1.0)),
                min_coverage=float(calibration_data.get("common_face_min_coverage", 0.9)))
            if common_face:
                result["inliers"] = common_face["inliers"]
                result["inlier_count"] = len(common_face["inliers"])
                result["max_inlier_residual_mm"] = max(
                    row["residual_mm"] for row in common_face["inliers"])

        anchors = [{"face_id": row["face_id"], "position": row["position"], "label_offset": [32, -26]}
                   for row in result["inliers"]]
        old_anchors = (old or {}).get("anchors", [])
        shifts = []
        shift_rows = []
        old_xy_residuals = []
        old_xy_rows = []
        if old_anchors:
            old_positions = np.asarray([a["position"] for a in old_anchors], dtype=np.float32)
            for point_row, anchor in zip(result["inliers"], anchors):
                pos = np.asarray(anchor["position"], dtype=np.float32)
                delta = np.sqrt(np.sum((old_positions - pos) ** 2, axis=1))
                closest = int(np.argmin(delta))
                shifts.append(float(delta[closest]))
                mapped = np.asarray(point_row["mapped_xy"], dtype=np.float32)
                old_xy = np.sqrt(np.sum((old_positions[:, :2] - mapped) ** 2, axis=1))
                xy_closest = int(np.argmin(old_xy))
                old_xy_residuals.append(float(old_xy[xy_closest]))
                old_xy_rows.append({"point": point_row["name"],
                                    "previous_xy_residual_mm": round(float(old_xy[xy_closest]), 3),
                                    "previous_face": old_anchors[xy_closest]["face_id"],
                                    "mapped_xy": mapped.tolist(),
                                    "previous_xyz": old_positions[xy_closest].tolist()})
                shift_rows.append({"point": point_row["name"], "shift_mm": round(float(delta[closest]), 3),
                                   "previous_face": old_anchors[closest]["face_id"],
                                   "new_face": anchor["face_id"],
                                   "previous_xyz": old_positions[closest].tolist(),
                                   "new_xyz": pos.tolist()})
        old_face_ids = set((old or {}).get("face_ids", []))
        changed_face_count = sum(a["face_id"] not in old_face_ids for a in anchors) if old else None
        source_pages = sorted({p for row in point_rows for p in row.get("pages", []) if p})
        transform_desc = transform.get("name", "calibrated MSOP→CAD transform")
        transform_formula = (
            f"x'={transform.get('x_sign', 1)}·x+{transform.get('x_offset_mm', 0)} mm; "
            f"y'={transform.get('y_sign', 1)}·y+{transform.get('y_offset_mm', 0)} mm"
        )
        correspondence_rows = [
            {"point": row["name"], "mapped_xy": [round(v, 2) for v in row["mapped_xy"]],
             "anchor_xy": [round(v, 2) for v in row["position"][:2]],
             "r_mm": round(row["residual_mm"], 2), "face": row["face_id"]}
            for row in result["inliers"][:10]
        ]
        note = (
            f"通用语料检索候选；检索方式：{retrieval}。来源页：{source_pages}。"
            f"坐标校准：{transform_desc}（{transform_formula}；该变换已应用到下列锚点）。"
            f"{result['inlier_count']}/{result['point_count']} 个显式点位"
            f"通过 {calibration_data.get('max_xy_residual_mm', 1.0):g} mm XY 检索门限，"
            f"最大内点残差 {result['max_inlier_residual_mm']:.3f} mm。"
            "逐点对应（source_xy→mapped_xy→CAD表面锚点）："
            + json.dumps(correspondence_rows, ensure_ascii=False, separators=(',', ':'))
            + (f"；另有{len(result['inliers']) - len(correspondence_rows)}个内点见原始Location Definitions"
               if len(result["inliers"]) > len(correspondence_rows) else "")
            + ("；点投影到候选三角面内，并将投影限制在原候选面集合，避免装配层间跳面。"
               if old and (args.recheck_pending or args.recheck_model_pending) else "；点位投影到 CAD 三角面内。")
            + (f"同一{'平面度' if common_group_reason == 'constructed_plane_flatness' else '源Z层级'}测量组联合匹配到 CAD 面 {common_face['face_id']}，"
               f"覆盖率 {common_face['coverage']:.1%}，组内中位 XY 残差 "
               f"{common_face['median_residual_mm']:.3f} mm，CAD 锚点 Z 范围 "
               f"{common_face['z_span_mm']:.3f} mm。"
               + (f"XY 几何仍无法区分同样覆盖点组的叠层候选面 "
                  f"{[row['face_id'] for row in common_face['alternatives']]}；"
                  "需复核 CAD/MSOP 的 Z 基准或实物面，不能视为已消除面选择歧义。"
                  if common_face["ambiguous"] else "")
               if common_face else "")
            + "MSOP Z 坐标尚未纳入坐标系配准；该 XY 残差仅用于候选检索，标记仍待复核，不表示满足计量要求。"
        )
        if not args.dry_run:
            payload = {"anchors": anchors, "status": "pending_review", "note": note,
                       "candidate_face_ids": candidate_face_ids or list(dict.fromkeys(a["face_id"] for a in anchors)),
                       "origin": old.get("origin") if old else "geometry_candidate",
                       "confidence": old.get("confidence") if old else None,
                       "cad_sha256": mesh["source_sha256"], "document_id": item["document_id"]}
            request = urllib.request.Request(
                f"{args.origin}/api/viewer/projects/{args.project}/fai/{item['number']}/annotation",
                data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                json.load(response)
        output.append({"fai": item["number"], "result": "candidate_ready" if args.dry_run else "candidate_created",
                       "retrieval": retrieval, "source_pages": source_pages,
                       "points": result["point_count"], "inliers": result["inlier_count"],
                       "inlier_ratio": round(result["inlier_ratio"], 3),
                       "median_residual_mm": round(float(np.median([r["residual_mm"] for r in result["projected"]])), 3),
                       "max_inlier_residual_mm": round(float(result["max_inlier_residual_mm"]), 3),
                       "unique_faces": len({a["face_id"] for a in anchors}),
                       "common_face": common_face["face_id"] if common_face else None,
                       "common_group_reason": common_group_reason if common_face else None,
                       "common_face_coverage": round(common_face["coverage"], 3) if common_face else None,
                       "common_face_median_residual_mm": round(common_face["median_residual_mm"], 4) if common_face else None,
                       "common_face_z_span_mm": round(common_face["z_span_mm"], 4) if common_face else None,
                       "common_face_ambiguous": common_face["ambiguous"] if common_face else None,
                       "common_face_alternatives": common_face["alternatives"] if common_face else [],
                       "median_shift_from_previous_mm": round(float(np.median(shifts)), 3) if shifts else None,
                       "max_shift_from_previous_mm": round(float(max(shifts)), 3) if shifts else None,
                       "largest_anchor_shifts": sorted(shift_rows, key=lambda row: row["shift_mm"], reverse=True)[:5],
                       "previous_xy_residual_median_mm": round(float(np.median(old_xy_residuals)), 3) if old_xy_residuals else None,
                       "previous_xy_residual_max_mm": round(float(max(old_xy_residuals)), 3) if old_xy_residuals else None,
                       "previous_points_over_1mm": sum(v > 1 for v in old_xy_residuals),
                       "largest_previous_xy_residuals": sorted(old_xy_rows, key=lambda row: row["previous_xy_residual_mm"], reverse=True)[:3],
                       "anchors_on_new_faces": changed_face_count,
                       "transform": transform_desc})

    print(json.dumps({"dry_run": args.dry_run, "results": output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
