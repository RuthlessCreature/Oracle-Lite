from __future__ import annotations
import json
from types import SimpleNamespace
import pymupdf
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from oracle_viewer.annotations import triangle_distance
from oracle_viewer.api import attach_viewer
from oracle_viewer.store import ViewerStore


@pytest.fixture
def setup(tmp_path):
    s = ViewerStore(tmp_path / "viewer")
    pid = s.create("annotation test", "613-13599", "06")["id"]
    source = tmp_path / "MSOP.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text(
            (20, 30), "MSOP: FAI 23 62.58 +0.12/-0.12\nCIR_N1_VISION"
        )
        pdf.new_page().insert_text(
            (20, 30), "MSOP: FAI 24 53.40 +0.12/-0.12\nCIR_N4_VISION"
        )
        pdf.save(source)
    doc = s.add_document(pid, "msop", source, source.name)
    mesh = {
        "source_sha256": "cad-a",
        "metrology_ready": False,
        "units": "mm",
        "faces": [
            {
                "id": "face-a",
                "face": 42,
                "body": 1,
                "instance_path": [1, 2],
                "positions": [0, 0, 0, 2, 0, 0, 0, 2, 0],
                "indices": [0, 1, 2],
                "center": [2 / 3, 2 / 3, 0],
            },
            {
                "id": "face-b",
                "face": 42,
                "body": 2,
                "instance_path": [1, 3],
                "positions": [20, 0, 0, 22, 0, 0, 20, 2, 0],
                "indices": [0, 1, 2],
                "center": [20 + 2 / 3, 2 / 3, 0],
            },
        ],
    }
    mesh_path = s._dir(pid) / "preview.mesh.json"
    mesh_path.write_text(json.dumps(mesh))
    data = s.get(pid)
    data["model"] = {
        "status": "preview",
        "source_sha256": "cad-a",
        "mesh_file": str(mesh_path),
        "metrology_ready": False,
    }
    s._write(data)
    return s, pid, doc


def save(s, pid, doc, **kwargs):
    options = {
        "anchors": [{"face_id": "face-a", "position": [0.5, 0.5, 0]}],
        "note": "specified edge",
        "status": "pending_review",
        "cad_sha256": "cad-a",
        "document_id": doc["id"],
    }
    options.update(kwargs)
    return s.annotate(pid, 23, **options)


def test_preview_annotation_persists_and_review_does_not_qualify_measurement(setup):
    s, pid, doc = setup
    a = save(s, pid, doc)
    assert a["metrology_ready"] is False and a["status"] == "pending_review"
    assert a["anchors"][0]["source_face_id"] == 42
    assert a["anchors"][0]["instance_path"] == [1, 2]
    assert ViewerStore(s.root).summary(s.get(pid))["marked_count"] == 0
    assert s.summary(s.get(pid))["located_count"] == 0
    assert s.summary(s.get(pid))["reviewed_count"] == 0
    save(s, pid, doc, status="reviewed")
    public = s.public_project(pid)
    assert public["reviewed_count"] == 1
    assert public["items"][0]["annotation"]["metrology_ready"] is False
    assert public["items"][0]["binding"] is None
    with pytest.raises(ValueError, match="FAI"):
        s.bind(pid, 23, ["face-a"], "visual pin cannot enable formal binding")


def test_automatic_marker_status_is_complete_without_claiming_manual_review(setup):
    s, pid, doc = setup
    annotation = save(s, pid, doc, status='auto_marked', origin='local_model_candidate', confidence='low')
    assert annotation['status'] == 'auto_marked'
    assert annotation['metrology_ready'] is False
    public = s.public_project(pid)
    assert public['marked_count'] == 1
    assert public['reviewed_count'] == 0
    assert public['items'][0]['annotation']['status'] == 'auto_marked'


def test_manual_binding_or_annotation_never_counts_as_an_automatic_fai_mark(setup):
    s, pid, doc = setup
    save(s, pid, doc)
    data=s.get(pid)
    detail=s.detail(pid,23)
    data.setdefault('bindings',{})[detail['item']['key']]=dict(face_ids=['face-a'],anchors=[[.5,.5,0]])
    s._write(data)
    summary=s.summary(s.get(pid))
    assert summary['marked_count']==0
    assert summary['located_count']==1  # legacy manual data is visible only as provenance


def test_large_source_group_retains_all_points_and_validates_each(setup):
    s,pid,doc=setup
    anchors=[dict(face_id='face-a',position=[.5,.5,0]) for _ in range(102)]
    assert len(save(s,pid,doc,anchors=anchors)['anchors'])==102
    anchors[-1]['position']=[.5,.5,.1]
    with pytest.raises(ValueError,match='not on'):
        save(s,pid,doc,anchors=anchors)


@pytest.mark.parametrize(
    "change",
    [
        {"cad_sha256": "other-cad"},
        {"document_id": "other-msop"},
        {"anchors": [{"face_id": "foreign-face", "position": [0.5, 0.5, 0]}]},
        {"anchors": [{"face_id": "face-b", "position": [0.5, 0.5, 0]}]},
        {"anchors": [{"face_id": "face-a", "position": [0.5, 0.5, 0.02]}]},
        {"anchors": [{"face_id": "face-a", "position": [float("nan"), 0.5, 0]}]},
        {"status": "qualified"},
    ],
)
def test_invalid_or_stale_annotations_are_rejected(setup, change):
    s, pid, doc = setup
    with pytest.raises(ValueError):
        save(s, pid, doc, **change)
    assert s.get(pid)["annotations"] == {}


def test_annotations_invalidated_when_msop_changes_and_history_retained(
    setup, tmp_path
):
    s, pid, doc = setup
    save(s, pid, doc)
    replacement = tmp_path / "replacement.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((20, 30), "MSOP: FAI 23 62.59 +0.12/-0.12")
        pdf.save(replacement)
    s.add_document(pid, "msop", replacement, replacement.name)
    data = s.get(pid)
    assert not data["annotations"]
    event = next(e for e in data["events"] if e["action"] == "annotations_invalidated")
    assert event["previous"][f"{doc['id']}:23"]["source_sha256"] == "cad-a"


def test_removal_and_task_snapshot(setup):
    s, pid, doc = setup
    save(s, pid, doc)
    task = s.task(pid, 23, "explain optical placement")
    assert task["annotation"]["status"] == "pending_review"
    assert "不是测量结果" in task["input_context"]
    assert task["binding"] is None
    s.remove_annotation(pid, 23)
    assert s.summary(s.get(pid))["marked_count"] == 0


def test_triangle_distance_interior_edges_and_degenerate_geometry():
    a, b, c = (0, 0, 0), (2, 0, 0), (0, 2, 0)
    assert triangle_distance((0.5, 0.5, 0.25), a, b, c) == pytest.approx(0.0625)
    assert triangle_distance((2, 2, 0), a, b, c) == pytest.approx(2)
    assert triangle_distance((1, 0, 0.5), a, b, (1, 0, 0)) == pytest.approx(0.25)
    assert triangle_distance((1, 0, 0), a, a, a) == pytest.approx(1)


def test_annotation_api_preview_export_review_and_bad_coordinate_types(setup, tmp_path):
    s, pid, doc = setup
    state = SimpleNamespace(
        cfg=SimpleNamespace(
            state_dir=tmp_path, training_output_dir=tmp_path / "training"
        )
    )
    app = FastAPI()
    attach_viewer(app, state)
    payload = {
        "anchors": [{"face_id": "face-a", "position": [0.5, 0.5, 0.0]}],
        "cad_sha256": "cad-a",
        "document_id": doc["id"],
    }
    path = f"/api/viewer/projects/{pid}/fai/23/annotation"
    with TestClient(app) as client:
        assert client.post(path, json=payload).status_code == 200
        r = client.get(f"/api/viewer/projects/{pid}/export")
        assert r.json()["items"][0]["annotation"]["anchors"][0]["position"] == [
            0.5,
            0.5,
            0.0,
        ]
        payload["status"] = "reviewed"
        assert client.post(path, json=payload).json()["metrology_ready"] is False
        payload["anchors"][0]["position"][0] = True
        assert client.post(path, json=payload).status_code == 422
        assert client.delete(path).status_code == 200
        assert client.get(f"/api/viewer/projects/{pid}").json()["marked_count"] == 0


def test_task_becomes_stale_for_annotation_changes_but_not_label_layout(setup):
    s, pid, doc = setup
    save(s, pid, doc)
    s.task(pid, 23, "optical plan")
    assert not s.public_project(pid)["tasks"][0]["stale"]
    save(s, pid, doc, anchors=[{"face_id": "face-a", "position": [0.5, 0.5, 0], "label_offset": [75, 20]}])
    assert not s.public_project(pid)["tasks"][0]["stale"]
    save(s, pid, doc, status="reviewed")
    assert s.public_project(pid)["tasks"][0]["stale"]
