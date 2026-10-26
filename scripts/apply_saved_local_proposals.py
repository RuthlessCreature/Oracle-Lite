"""Safely apply validated face selections from the latest local-model run."""
from __future__ import annotations
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oracle_viewer.store import ViewerStore


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--origin", default="http://127.0.0.1:7870")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-support", type=float, default=0.65,
                    help="minimum explicit-point inlier fraction for applying a display suggestion")
    args = ap.parse_args()
    store = ViewerStore(ROOT / "talker/state/viewer")
    project = store.get(args.project)
    run = project.get("local_model_marking") or {}
    if run.get("status") != "completed":
        raise SystemExit("latest local-model run is not complete")
    results = []
    for row in run.get("items", []):
        decision = re.sub(r"[^A-Z_]", "", str(row.get("decision") or "").upper())
        if decision not in {"PROPOSE_CANDIDATE", "PROPOSE_REVIEW", "PROPOSE_LOW_CONF",
                            "PENDING_REVIEW", "REVIEW_REQUIRED"}:
            continue
        if row.get("truncated"):
            results.append({"fai": row.get("fai"), "result": "skipped_truncated_output"})
            continue
        try:
            parsed = json.loads(row.get("raw_output", ""))
        except json.JSONDecodeError:
            results.append({"fai": row.get("fai"), "result": "skipped_invalid_json"})
            continue
        number = int(re.sub(r"\D", "", str(row.get("fai") or parsed.get("fai") or "")))
        item = next((i for i in store.items(project) if i["number"] == number), None)
        if not item or not item.get("annotation") or item["annotation"].get("origin") != "geometry_candidate":
            results.append({"fai": number, "result": "skipped_candidate_changed"})
            continue
        old = item["annotation"]
        support = re.search(r"(\d+)/(\d+) 个显式点位", old.get("note", ""))
        support_ratio = (int(support.group(1)) / int(support.group(2))) if support and int(support.group(2)) else 1.0
        if support_ratio < args.min_support:
            results.append({"fai": number, "result": "skipped_weak_geometric_support",
                            "inlier_ratio": round(support_ratio, 3),
                            "required": args.min_support})
            continue
        available = {a["face_id"] for a in old.get("anchors", [])}
        selected = parsed.get("face_ids")
        if not isinstance(selected, list) or not selected or any(
                not isinstance(face_id, str) or face_id not in available for face_id in selected):
            results.append({"fai": number, "result": "skipped_invalid_face_selection"})
            continue
        selected = list(dict.fromkeys(selected))
        anchors = [a for a in old["anchors"] if a["face_id"] in selected]
        if not anchors:
            results.append({"fai": number, "result": "skipped_empty_anchors"})
            continue
        confidence = str(parsed.get("confidence", "low")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        note = (
            f"本地模型建议，待人工复核；原始模型决策 {decision}，置信度 {confidence}（未经校准）。"
            f"模型证据：{parsed.get('evidence', '未提供')}。候选来源：{old.get('note', '')}"
        )
        if not args.dry_run:
            payload = {"anchors": anchors, "status": "pending_review", "note": note,
                       "origin": "local_model_candidate", "confidence": confidence,
                       "cad_sha256": old["source_sha256"], "document_id": old["document_id"]}
            request = urllib.request.Request(
                f"{args.origin}/api/viewer/projects/{args.project}/fai/{number}/annotation",
                data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                json.load(response)
        results.append({"fai": number, "result": "would_apply" if args.dry_run else "applied",
                        "decision": decision, "confidence": confidence,
                        "face_count": len(selected), "anchor_count": len(anchors)})
    print(json.dumps({"dry_run": args.dry_run, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
