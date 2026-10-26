"""Add reviewed-source visual candidates for this exact D52 CAD/MSOP pair.

These are initial visual correspondences for FAI23/24, not automatic mapping of
all FAI items. Existing user annotations are preserved. No candidate is reviewed
or metrologically qualified by this script.
"""

from __future__ import annotations
import argparse
import json
import urllib.request

CAD_SHA256 = "ec5e3a7adacd605ebe88958a9aa329e819e26a23ffbad6798a244b59a6385b40"
CANDIDATES = {
    23: {
        "feature": "CIR_N1_VISION",
        "page": 33,
        "faces": [(59, 227, "min"), (60, 195, "max")],
    },
    24: {"feature": "CIR_N4_VISION", "page": 34, "faces": [(17, 3144, "max")]},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--origin", default="http://127.0.0.1:7870")
    args = parser.parse_args()
    base = args.origin.rstrip("/") + "/api/viewer/projects/" + args.project

    def get(path):
        return json.load(urllib.request.urlopen(base + path))

    project = get("")
    if (
        project["drawing_no"] != "613-13599"
        or project["revision"] != "06"
        or project["model"]["source_sha256"] != CAD_SHA256
    ):
        raise ValueError(
            "Candidate correspondence is scoped to the exact D52 Rev06 CAD"
        )
    mesh = get("/mesh")
    by_key = {(f["body"], f["face"]): f for f in mesh["faces"]}
    for n, candidate in CANDIDATES.items():
        item = next(i for i in project["items"] if i["number"] == n)
        detail = get("/fai/" + str(n))
        if (
            candidate["page"] not in item["pages"]
            or candidate["feature"] not in detail["context"]
        ):
            raise ValueError("MSOP feature/page evidence does not match candidate")
        if item.get("annotation") or item.get("binding"):
            print(f"FAI {n}: existing position preserved")
            continue
        anchors = []
        for body, fid, side in candidate["faces"]:
            face = by_key[(body, fid)]
            vertices = [
                face["positions"][i : i + 3]
                for i in range(0, len(face["positions"]), 3)
            ]
            top = max(v[2] for v in vertices)
            rim = [v for v in vertices if abs(v[2] - top) < 1e-5]
            point = (min if side == "min" else max)(rim, key=lambda v: v[0])
            anchors.append(
                {
                    "face_id": face["id"],
                    "position": point,
                    "label_offset": [45, -42] if n == 23 else [52, 32],
                }
            )
        payload = {
            "anchors": anchors,
            "cad_sha256": CAD_SHA256,
            "document_id": item["document_id"],
            "status": "pending_review",
            "origin": "msop_visual_candidate",
            "note": f"MSOP 第 {candidate['page']} 页 {candidate['feature']} 的视觉对应候选：依据细节图的孔口形态与装配相对位置核对。锚点是 CAD 显示孔边上的引线位置，不是圆心测量值；请在对照标注视图中复核。",
        }
        request = urllib.request.Request(
            base + f"/fai/{n}/annotation",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        result = json.load(urllib.request.urlopen(request))
        print(f"FAI {n}: {len(result['anchors'])} anchors, {result['status']}")


if __name__ == "__main__":
    main()
