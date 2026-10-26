from types import SimpleNamespace

from scripts import run_local_fai_marking
from scripts.run_local_fai_marking import configured_viewer_model
from oracle_viewer import marking_pipeline


def test_marking_cli_reuses_talker_model_when_viewer_endpoint_is_not_split(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("ORACLE_VIEWER_MODEL_BASE_URL", raising=False)
    monkeypatch.setenv("ORACLE_TALKER_OPENAI_BASE_URL", "http://127.0.0.1:8081")
    monkeypatch.setenv("ORACLE_TALKER_OPENAI_MODEL", "strata-shared")
    monkeypatch.setenv("ORACLE_TALKER_INFERENCE_BACKEND", "strata-native")

    model = configured_viewer_model(SimpleNamespace(training_output_dir=tmp_path))

    assert model.openai_base_url == "http://127.0.0.1:8081"
    assert model.openai_model == "strata-shared"
    assert model.openai_backend_label == "strata-native"


def test_marking_cli_uses_the_separate_viewer_model_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("ORACLE_VIEWER_MODEL_BASE_URL", "http://127.0.0.1:8091/")
    monkeypatch.setenv("ORACLE_VIEWER_MODEL", "viewer-model")
    monkeypatch.setenv("ORACLE_VIEWER_MODEL_BACKEND", "strata-native")

    model = configured_viewer_model(SimpleNamespace(training_output_dir=tmp_path))

    assert model.openai_base_url == "http://127.0.0.1:8091"
    assert model.openai_model == "viewer-model"
    assert model.openai_backend_label == "strata-native"


def test_marking_cli_finalizes_viewer_coverage_from_saved_annotations(monkeypatch):
    class Store:
        updates = None
        def get(self, project_id): return {"id": project_id}
        def items(self, project):
            return [
                {"number": 1, "annotation": {"origin": "local_model_candidate", "status": "auto_marked",
                                               "anchors": [{"face_id": "face-a", "position": [0, 0, 0]}]}},
                {"number": 2, "annotation": None},
            ]
        def update_local_model_marking(self, project_id, **changes): self.updates = changes

    monkeypatch.setattr(run_local_fai_marking, "unmarked_diagnostics",
                        lambda _store, _project: [{"fai": 2, "result": "no_geometry_candidate"}])
    store = Store()
    coverage = run_local_fai_marking.finalize_marking_status(
        store, "project-1", {"items": []},
        {"counts": {"accepted": 1, "abstain": 1, "skipped": 0, "processed": 2},
         "items": [{"fai": 2, "result": "abstain_or_invalid"}]},
        {"model_alias": "local-strata", "inference_backend": "strata-native"},
    )
    assert coverage["marked"] == 1
    assert store.updates["status"] == "partial"
    assert store.updates["message"] == "已标记 1/2 项；未标记 1 项"
    assert store.updates["counts"]["processed"] == 2
    assert store.updates["items"][0]["unmarked_diagnostic"]["result"] == "no_geometry_candidate"


def test_marking_cli_records_retryable_failure_without_clearing_saved_marks():
    class Store:
        annotations = {"msop:1": {"origin": "local_model_candidate", "status": "auto_marked"}}
        updates = None

        def update_local_model_marking(self, project_id, **changes):
            assert project_id == "project-1"
            self.updates = changes

    store = Store()
    run_local_fai_marking.record_pipeline_failure(
        store, "project-1", "local_model_marking", ConnectionError("Strata unavailable")
    )

    assert store.updates["status"] == "failed"
    assert store.updates["fingerprint"] is None
    assert "ConnectionError" in store.updates["error"]
    assert store.updates["progress"]["stage"] == "local_model_marking"
    assert store.annotations["msop:1"]["status"] == "auto_marked"


def test_registration_failure_keeps_all_fai_in_the_local_model_visual_queue(monkeypatch):
    class Store:
        refreshed = []
        def refresh_msop_index(self, project_id): self.refreshed.append(project_id)

    store=Store();progress=[]
    monkeypatch.setattr(marking_pipeline,'ensure_registration',
        lambda *_args: (_ for _ in ()).throw(ConnectionError('Strata unavailable')))
    monkeypatch.setattr(marking_pipeline,'build_candidates',lambda *_args,**_kwargs:{
        'items':[{'fai':i,'result':'local_visual_candidate_ready'} for i in range(1,83)],
        'candidates':[{'fai':i,'candidate_kind':'model_visual_localization','anchors':[]} for i in range(1,83)],
    })
    registration,generated=marking_pipeline.prepare_marking_candidates(
        store,'p',object(),progress=progress.append)
    assert registration is None
    assert store.refreshed==['p']
    assert len(generated['items'])==len(generated['candidates'])==82
    assert generated['registration_error'].startswith('ConnectionError:')
    assert progress[0]['stage']=='visual_fallback'
