from threading import Lock

from oracle_viewer.local_matching import (
    _local_model_runtime_ready,
    _validate_coordinate_variant_choices,
    mark_geometry_candidates,
)


class _Store:
    def get(self, _project_id):
        return {"drawing_no": "local-test", "revision": "A", "documents": []}

    def mesh(self, _project_id):
        return {"source_sha256": "cad-hash", "faces": []}

    def active_msop(self, _project):
        return {"method_pages": []}

    def items(self, _project):
        return []


class _LoopbackStrata:
    loaded = True
    bundle = None
    openai_base_url = "http://127.0.0.1:8081"
    _lock = Lock()

    def info(self):
        return {"inference_backend": "strata-native", "model_alias": "local-strata"}


def test_loaded_loopback_strata_is_ready_without_transformers_bundle():
    model = _LoopbackStrata()

    assert _local_model_runtime_ready(model)
    result = mark_geometry_candidates(_Store(), model, None, "p", generated_candidates=[])
    assert result["inference_backend"] == "strata-native"
    assert result["counts"]["processed"] == 0


def test_direct_transformers_backend_requires_loaded_bundle():
    class MissingBundle:
        loaded = True
        bundle = None
        openai_base_url = ""

    assert not _local_model_runtime_ready(MissingBundle())


def test_local_model_coordinate_choice_must_match_one_source_occurrence_exactly():
    points=[dict(name='DATUMC_PNT_P04',coordinate_variants=[
        dict(page=13,program='UP',coordinate_system='ALI_ABC_A',xyz=[-18.57,129.22,-.35]),
        dict(page=75,program='DOWN',coordinate_system='ALI_ABC_A',xyz=[-18.57,64.52,-.35]),
    ])]
    valid=[dict(name='DATUMC_PNT_P04',page=75,program='DOWN',coordinate_system='ALI_ABC_A',
                xyz=[-18.57,64.52,-.35])]
    ok,selected=_validate_coordinate_variant_choices(points,valid)
    assert ok and selected[0]['page']==75
    assert not _validate_coordinate_variant_choices(points,[])[0]
    invalid=[dict(name='DATUMC_PNT_P04',page=75,program='DOWN',coordinate_system='ALI_ABC_A',
                  xyz=[-18.57,65.0,-.35])]
    assert not _validate_coordinate_variant_choices(points,invalid)[0]
