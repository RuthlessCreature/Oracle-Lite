from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from talker.attachments import parse_attachment
from talker.config import load_config
from talker.history import ChatHistory
from talker.resolver import discover_latest_adapter


def test_talker_config_has_single_training_output_field(tmp_path: Path):
    cfg_file = tmp_path / "talker.yaml"
    cfg_file.write_text('training_output_dir: "./training-root"\n', encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.training_output_dir == (tmp_path / "training-root").resolve()
    assert cfg.history_db.name == "history.sqlite3"
    assert cfg.uploads_dir.exists()
    assert cfg.parsed_dir.exists()


def test_talker_config_rejects_unknown_keys(tmp_path: Path):
    cfg_file = tmp_path / "talker.yaml"
    cfg_file.write_text(
        'training_output_dir: "./x"\nport: 9999\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only training_output_dir"):
        load_config(cfg_file)


def test_chat_history_persists_messages_and_attachments(tmp_path: Path):
    db = tmp_path / "history.sqlite3"
    history = ChatHistory(db)
    conversation = history.new_conversation()
    user_id = history.add_message(conversation["id"], "user", "check drawing")
    history.add_attachment(
        user_id,
        filename="part.step",
        stored_path="/tmp/part.step",
        mime_type="application/octet-stream",
        parsed_text="Entity count: 10",
        images=[],
    )
    history.add_message(conversation["id"], "assistant", "ok")

    reopened = ChatHistory(db)
    messages = reopened.get_messages(conversation["id"])
    assert [row["role"] for row in messages] == ["user", "assistant"]
    assert messages[0]["attachments"][0]["filename"] == "part.step"
    assert messages[0]["attachments"][0]["parsed_text"] == "Entity count: 10"


def test_adapter_resolver_finds_local_base_and_latest_adapter(tmp_path: Path):
    root = tmp_path / "output"
    base = root / "models" / "Qwen3.5-9B-Base"
    base.mkdir(parents=True)
    (base / "config.json").write_text("{}", encoding="utf-8")

    run1 = root / "_corpora" / "a" / "training" / "snap1" / "run"
    final = run1 / "adapter-final"
    final.mkdir(parents=True)
    (final / "adapter_config.json").write_text("{}", encoding="utf-8")
    (run1 / "run.json").write_text(
        json.dumps({"base_model_path": str(base)}),
        encoding="utf-8",
    )

    run2 = root / "_corpora" / "a" / "training" / "snap2" / "run"
    checkpoint = run2 / "checkpoint-7"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (run2 / "run.json").write_text(
        json.dumps({"base_model_path": str(base)}),
        encoding="utf-8",
    )

    os.utime(final / "adapter_config.json", (1000, 1000))
    os.utime(checkpoint / "adapter_config.json", (2000, 2000))

    bundle = discover_latest_adapter(root)
    assert bundle.adapter_dir == final.resolve()
    assert bundle.base_model_path == base.resolve()
    assert bundle.source_kind == "final"


def test_corpus_retrieval_preserves_datum_and_fai_identifiers(tmp_path: Path):
    from oracle_lite.canonical import CanonicalDocument, CanonicalSegment
    from oracle_lite.db import Registry
    from talker.retrieval import CorpusRetriever

    root = tmp_path / "output"
    canonical = root / "_corpora" / "corpus-a" / "canonical" / "msop.json"
    CanonicalDocument(
        content_hash="fixture-hash",
        source_path="fixture-MSOP.pdf",
        source_type="pdf",
        parser_name="fixture",
        parser_version="1",
        text="",
        segments=[
            CanonicalSegment(
                text=(
                    "Measured System\nDATUM A\nALI_ITER\nMASTER_P5\n"
                    "FAI 7 (SPC C) 0 +0.15/0.00"
                ),
                metadata={"page": 22},
            ),
            CanonicalSegment(
                text="DATUM B\nALI_ABC_A\nMASTER_P3_SHANK",
                metadata={"page": 23},
            ),
        ],
    ).write_json(canonical)

    retriever = CorpusRetriever(root, tmp_path / "retrieval.sqlite3")
    datum = retriever.search("基准 A 使用什么程序和探针？", limit=2)
    fai = retriever.search("FAI 7 测量什么？", limit=2)

    assert datum[0].section == "page 22"
    assert "ALI_ITER" in datum[0].text
    assert fai[0].section == "page 22"
    assert "FAI 7" in fai[0].text

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    qa_path = corpus / "qa_pairs.jsonl"
    qa_records = [
        {
            "question": "MSOP 中 DATUM A 的 Probe 和 Program 分别是什么？",
            "answer": "Probe 是 MASTER_P5，Program 是 UP。",
            "context": "Probe MASTER_P5; Program UP",
            "source_path": "fixture-MSOP.pdf, page 9",
        },
        {
            "question": "MSOP 中 DATUM B 的 Probe 和 Program 分别是什么？",
            "answer": "Probe 是 MASTER_P3_SHANK，Program 是 UP。",
            "context": "Probe MASTER_P3_SHANK; Program UP",
            "source_path": "fixture-MSOP.pdf, page 11",
        },
    ]
    qa_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in qa_records),
        encoding="utf-8",
    )
    registry = Registry(root / "_corpora" / "corpus-a" / "_state" / "registry.sqlite3")
    registry.upsert_source(
        root=corpus,
        path=qa_path,
        size=qa_path.stat().st_size,
        mtime_ns=qa_path.stat().st_mtime_ns,
        content_hash="qa-hash",
    )
    curated = retriever.curated_context("DATUM A 用什么程序和探针？")
    assert curated is not None
    assert "MASTER_P5" in curated
    assert "page 9" in curated
    assert retriever.curated_context("DATUM B 用什么程序和探针？") is not None
    followup = retriever.curated_context(
        "DATUM B 呢？",
        context_query="MSOP 中 DATUM A 的 Probe 和 Program 分别是什么？",
    )
    assert followup is not None
    assert "MASTER_P3_SHANK" in followup
    assert "page 11" in followup


def test_corpus_retrieval_matches_arbitrary_letter_number_identifiers(tmp_path: Path):
    from oracle_lite.canonical import CanonicalDocument, CanonicalSegment
    from talker.retrieval import CorpusRetriever

    root = tmp_path / "output"
    canonical = root / "_corpora" / "corpus-a" / "canonical" / "spec.json"
    CanonicalDocument(
        content_hash="generic-id-hash",
        source_path="pressure-spec.pdf",
        source_type="pdf",
        parser_name="fixture",
        parser_version="1",
        text="",
        segments=[
            CanonicalSegment(
                text="Revision B pressure limit is 12 kPa.",
                metadata={"page": 4},
            ),
            CanonicalSegment(
                text="Revision C pressure limit is 15 kPa.",
                metadata={"page": 5},
            ),
        ],
    ).write_json(canonical)

    retriever = CorpusRetriever(root, tmp_path / "generic-retrieval.sqlite3")
    results = retriever.search("What is the pressure limit for Revision B?", limit=2)

    assert results[0].section == "page 4"
    assert "12 kPa" in results[0].text


def test_corpus_retrieval_is_scoped_to_selected_corpus(tmp_path: Path):
    from oracle_lite.canonical import CanonicalDocument, CanonicalSegment
    from talker.retrieval import CorpusRetriever

    root = tmp_path / "output"
    for corpus_id, source, answer in (
        ("corpus-a", "source-a.pdf", "Pressure limit is 12 kPa."),
        ("corpus-b", "source-b.pdf", "Pressure limit is 90 kPa."),
    ):
        CanonicalDocument(
            content_hash=f"hash-{corpus_id}",
            source_path=source,
            source_type="pdf",
            parser_name="fixture",
            parser_version="1",
            text="",
            segments=[CanonicalSegment(text=answer, metadata={"page": 1})],
        ).write_json(root / "_corpora" / corpus_id / "canonical" / "spec.json")

    retriever = CorpusRetriever(
        root,
        tmp_path / "scoped-retrieval.sqlite3",
        corpus_id="corpus-a",
    )
    results = retriever.search("pressure limit", limit=4)

    assert len(results) == 1
    assert results[0].source == "source-a.pdf"
    assert "12 kPa" in results[0].text


def test_attachment_parser_handles_text_and_step(tmp_path: Path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("bearing preload 12 N", encoding="utf-8")
    text, images = parse_attachment(text_file, tmp_path / "assets-text")
    assert "bearing preload 12 N" in text
    assert images == []

    step = tmp_path / "part.step"
    step.write_text(
        """ISO-10303-21;
HEADER;
FILE_SCHEMA (('AP214'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=ADVANCED_FACE('',(),$,.T.);
ENDSEC;
END-ISO-10303-21;
""",
        encoding="utf-8",
    )
    text, images = parse_attachment(step, tmp_path / "assets-step")
    assert "AP214" in text
    assert "CARTESIAN_POINT" in text
    assert images == []


def test_talker_web_asset_contains_chat_history_and_upload_controls():
    html = Path("talker/static/index.html").read_text(encoding="utf-8")
    assert 'id="conversations"' in html
    assert 'id="files"' in html
    assert 'id="sendBtn"' in html
    assert "/api/chat" in html
    assert "/api/reload-model" in html


def test_talker_entrypoint_and_versions():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    oracle_version = Path("oracle_lite/__init__.py").read_text(encoding="utf-8")
    talker_version = Path("talker/__init__.py").read_text(encoding="utf-8")
    assert 'version = "0.6.2"' in pyproject
    assert 'oracle-talker = "talker.__main__:main"' in pyproject
    assert 'include = ["oracle_lite*", "talker*", "oracle_viewer*"]' in pyproject
    assert '__version__ = "0.6.2"' in oracle_version
    assert '__version__ = "0.6.2"' in talker_version


def test_long_answers_continue_without_total_token_cap(monkeypatch, tmp_path: Path):
    from talker.model import TalkerModel

    talker = TalkerModel(tmp_path)
    calls = []

    def fake_generate(messages, **kwargs):
        calls.append(kwargs)
        index = len(calls)
        if index < 5:
            return f"FAI{index:02d}\n", True, 768
        return "FAI05 complete", False, 7

    monkeypatch.setattr(talker, "_generate_once", fake_generate)
    answer = talker._answer_with_auto_continue(
        [{"role": "user", "content": "列出所有FAI", "attachments": []}],
        max_context_chars=1000,
        chunk_tokens=768,
        max_images=0,
    )
    assert "FAI01" in answer
    assert "FAI05 complete" in answer
    assert len(calls) == 5
    assert "total_tokens" not in calls[0]


def test_unbounded_continuation_stops_on_repeated_chunk(monkeypatch, tmp_path: Path):
    from talker.model import TalkerModel

    talker = TalkerModel(tmp_path)
    calls = []

    def fake_generate(messages, **kwargs):
        calls.append(kwargs)
        return "same repeated chunk", True, 768

    monkeypatch.setattr(talker, "_generate_once", fake_generate)
    answer = talker._answer_with_auto_continue(
        [{"role": "user", "content": "继续", "attachments": []}],
        max_context_chars=1000,
        chunk_tokens=768,
        max_images=0,
    )
    assert answer == "same repeated chunk"
    assert len(calls) == 2


def test_talker_returns_final_answer_without_tagged_reasoning(monkeypatch, tmp_path):
    from talker.model import TalkerModel
    model = TalkerModel(tmp_path)
    chunks = iter([('<think>internal ', True, 4), ('reasoning</think>Pressure: 12 kPa', False, 5)])
    monkeypatch.setattr(model, '_generate_once', lambda *args, **kwargs: next(chunks))
    answer = model._answer_with_auto_continue([], max_context_chars=1000, chunk_tokens=768, max_images=0)
    assert answer == 'Pressure: 12 kPa'


def test_retrieved_images_match_source_and_page_and_are_deduplicated(tmp_path):
    from oracle_lite.canonical import CanonicalDocument, CanonicalSegment
    from talker.retrieval import CorpusRetriever
    from PIL import Image

    first = tmp_path / 'page1.png'
    second = tmp_path / 'page2.png'
    Image.new('RGB', (16, 16), 'white').save(first)
    Image.new('RGB', (16, 16), 'white').save(second)
    root = tmp_path / 'output'
    canonical = root / '_corpora' / 'a' / 'canonical' / 'spec.json'
    CanonicalDocument(content_hash='hash', source_path='spec.pdf', source_type='pdf',
                      parser_name='fixture', parser_version='1', text='', segments=[
                          CanonicalSegment(text='Revision B pressure limit 12 kPa', images=[str(first)], metadata={'page': 1}),
                          CanonicalSegment(text='Revision C pressure limit 15 kPa', images=[str(second)], metadata={'page': 2}),
                      ]).write_json(canonical)
    retriever = CorpusRetriever(root, tmp_path / 'fts.sqlite3', corpus_id='a')
    attachments = retriever.visual_context('Revision B pressure limit')
    assert len(attachments) == 1
    assert attachments[0]['images'] == [str(first)]
    assert attachments[0]['filename'] == 'spec.pdf, page 1'


def test_context_includes_fields_beyond_the_ranked_window_when_page_fits(tmp_path):
    from oracle_lite.canonical import CanonicalDocument, CanonicalSegment
    from talker.retrieval import CorpusRetriever
    root = tmp_path / 'output'
    canonical = root / '_corpora' / 'a' / 'canonical' / 'table.json'
    text = 'Pressure 12 kPa\n' + ('irrelevant row\n' * 95) + 'Temperature 90 C'
    CanonicalDocument(content_hash='hash', source_path='table.pdf', source_type='pdf',
                      parser_name='fixture', parser_version='1', text='', segments=[
                          CanonicalSegment(text=text, metadata={'page': 1})]).write_json(canonical)
    r = CorpusRetriever(root, tmp_path / 'fts.sqlite3', corpus_id='a')
    assert 'Temperature 90 C' not in r.search('Pressure limit', limit=1)[0].text
    context = r.context('Pressure limit', limit=1, max_chars=4000)
    assert 'Pressure 12 kPa' in context and 'Temperature 90 C' in context


def test_comprehensive_pdf_context_uses_positioned_blocks_and_keeps_requested_ids(tmp_path):
    from oracle_lite.canonical import CanonicalDocument, CanonicalSegment
    from talker.retrieval import CorpusRetriever

    root = tmp_path / 'output'
    canonical = root / '_corpora' / 'a' / 'canonical' / 'drawing.json'
    CanonicalDocument(content_hash='hash', source_path='drawing.pdf', source_type='pdf',
                      parser_name='fixture', parser_version='1', text='', segments=[
        CanonicalSegment(text='FAI 54\n8 spots per nut; max weld height 0.05',
                         metadata={'page': 7, 'positioned_blocks': [
                             {'x': 10.0, 'y': 20.0, 'kind': 'identifier', 'text': 'FAI 54'},
                             {'x': 12.0, 'y': 22.0, 'kind': 'text', 'text': '8 spots per nut; max weld height 0.05'},
                         ]}),
        CanonicalSegment(text='FAI 61-79 are weld spot heights; cannot measure by 443DZ',
                         metadata={'page': 103}),
    ]).write_json(canonical)
    retriever = CorpusRetriever(root, tmp_path / 'fts.sqlite3', corpus_id='a')

    details = retriever.context('List all weld height details', limit=1)
    assert '8 spots per nut; max weld height 0.05' in details
    assert 'FAI 54' not in details
    assert "Directly printed measurement callouts" in details
    assert 'FAI 61-79' not in details

    identifier = retriever.context('What is FAI 54?', limit=1)
    assert 'FAI 54' in identifier
    assert 'Positioned PDF text blocks' not in identifier


def test_explicit_complete_list_intent_adds_scoped_enumeration_instruction(monkeypatch, tmp_path):
    from talker.model import TalkerModel

    class Retriever:
        def curated_context(self, *args, **kwargs):
            return None
        def context(self, query, **kwargs):
            return '[Source 1: drawing.pdf, page 7]\n8 spots per nut; max height 0.05'
        def visual_context(self, *args, **kwargs):
            return []

    model = TalkerModel(tmp_path)
    model.model = object()
    model.processor = object()
    model.retriever = Retriever()
    seen = []
    monkeypatch.setattr(model, '_answer_with_auto_continue',
                        lambda messages, **kwargs: seen.append(messages) or 'answer')

    assert model.answer([{'role': 'user', 'content': 'how many weld spots?'}]) == 'answer'
    assert 'every directly matching source row or callout' in seen[-1][-1]['content']
    assert model.answer([{'role': 'user', 'content': 'what is the fixture weight?'}]) == 'answer'
    assert 'every directly matching source row or callout' not in seen[-1][-1]['content']


def test_narrow_multi_source_scope_question_gets_a_second_pass(monkeypatch, tmp_path):
    from talker.model import TalkerModel

    class Retriever:
        def curated_context(self, *args, **kwargs):
            return None
        def context(self, query, **kwargs):
            return ('[Source 1: procedure.pdf, page 9]\nProgram UP; Probe P5\n\n'
                    '[Source 2: procedure.pdf, page 73]\nProgram DOWN; Probe not stated')
        def visual_context(self, *args, **kwargs):
            return []

    model = TalkerModel(tmp_path)
    model.model = object()
    model.processor = object()
    model.retriever = Retriever()
    calls = []
    def generate(messages, **kwargs):
        calls.append(messages)
        return 'draft' if len(calls) == 1 else 'scoped correction'
    monkeypatch.setattr(model, '_answer_with_auto_continue', generate)

    answer = model.answer([{'role': 'user', 'content': '基准 A 使用什么程序和探针？'}])
    assert answer == 'scoped correction'
    assert len(calls) == 2
    assert 'Recheck the draft' in calls[1][-1]['content']


def test_numbered_spec_value_question_gets_review_only_with_multiple_sources():
    from talker.model import TalkerModel

    evidence = ('[Source 1: procedure.pdf, page 61]\nFAI 53 nominal 0.00\n'
                '[Source 2: procedure.pdf, page 62]\nFAI 53 tolerance ±0.03')
    assert TalkerModel._needs_scope_review('what is FAI 53 and its value?', evidence)
    assert TalkerModel._needs_scope_review('FAI 53 的数值和公差是多少？', evidence)
    assert not TalkerModel._needs_scope_review(
        'what is FAI 53 and its value?', '[Source 1: procedure.pdf, page 61]'
    )
    assert not TalkerModel._needs_scope_review(
        'list every FAI value', evidence
    )


def test_attachment_budget_keeps_question_and_marks_missing_complete_rows(tmp_path):
    from talker.model import TalkerModel
    model = TalkerModel(tmp_path)
    question = 'What is the pressure limit?'
    text = 'Pressure 12 kPa\n' + 'Temperature 90 C\n' * 100
    context, _ = model._context([{'role': 'user', 'content': question, 'attachments': [{
        'filename': 'spec.pdf', 'parsed_text': text, 'images': []}]}], 200)
    assert question in context
    assert 'Pressure 12 kPa' in context
    assert 'Attachment excerpt incomplete' in context
    assert len(context) <= 200


def test_explicit_viewer_image_budget_keeps_method_coordinates_drawing_and_cad(tmp_path):
    from talker.model import TalkerModel
    model=TalkerModel(tmp_path)
    images=[]
    for name in ('msop-method.png','msop-coordinates.png','2d-drawing.png','cad-atlas.png'):
        path=tmp_path/name
        path.write_bytes(b'image-placeholder-for-context-selection')
        images.append(str(path))
    messages=[{'role':'user','content':'Verify all four evidence images.',
               'attachments':[{'images':images}]}]
    _,selected=model._context(messages,1000,max_images=4)
    assert selected==images
    _,default=model._context(messages,1000)
    assert default==images[-3:]
    _,none=model._context(messages,1000,max_images=0)
    assert none==[]


def test_generate_once_passes_requested_image_count_through_context_collection(tmp_path,monkeypatch):
    from talker.model import TalkerModel
    model=TalkerModel(tmp_path,openai_base_url='http://127.0.0.1:8081',
                      openai_model='local-test',backend_label='strata-native')
    images=[]
    for i in range(4):
        path=tmp_path/f'image-{i}.png';path.write_bytes(b'context-test');images.append(str(path))
    captured={}
    def remote(context,paths,*args,**kwargs):
        captured['images']=paths
        return '{}',False,10
    monkeypatch.setattr(model,'_generate_openai',remote)
    model._generate_once([{'role':'user','content':'Review original evidence.',
        'attachments':[{'images':images}]}],max_context_chars=1000,max_new_tokens=128,max_images=4)
    assert captured['images']==images


def test_talker_no_longer_has_384_token_hard_cut_and_supports_lan_access():
    model = Path("talker/model.py").read_text(encoding="utf-8")
    main = Path("talker/__main__.py").read_text(encoding="utf-8")
    assert "GENERATION_TOTAL_TOKENS" not in model
    assert "GENERATION_FALLBACK_TOTAL_TOKENS" not in model
    assert "GENERATION_CHUNK_TOKENS = 768" in model
    assert "CONTINUATION_TAIL_CHARS = 8_000" in model
    assert 'max_new_tokens=384' not in model
    assert 'host="0.0.0.0"' in main
    assert 'sock.bind(("0.0.0.0", port))' in main
    assert "_lan_ipv4_addresses" in main


def test_reviewed_qa_is_evidence_and_latest_turn_controls_retrieval(tmp_path, monkeypatch):
    from talker.model import TalkerModel

    model = TalkerModel(tmp_path)
    model.model = object()
    model.processor = object()
    queries = []

    class FakeRetriever:
        def curated_context(self, query, **kwargs):
            return 'Reviewed QA Source: pressure spec; answer 12 kPa'

        def context(self, query, **kwargs):
            queries.append(query)
            return 'Source excerpt: Revision C pressure limit 15 kPa'

    model.retriever = FakeRetriever()

    def generate(messages, **kwargs):
        evidence = messages[-1]['attachments'][-1]['parsed_text']
        assert '12 kPa' in evidence and '15 kPa' in evidence
        return 'Model-generated answer for Revision C'

    monkeypatch.setattr(model, '_answer_with_auto_continue', generate)
    messages = [{'role': 'user', 'content': 'Revision B pressure limit?'},
                {'role': 'assistant', 'content': '12 kPa'},
                {'role': 'user', 'content': 'Revision C?'}]
    assert model.answer(messages) == 'Model-generated answer for Revision C'
    assert queries == ['Revision C?']
    assert 'attachments' not in messages[-1]


def test_deleted_registry_document_does_not_reappear_from_canonical_cache(tmp_path):
    from oracle_lite.canonical import CanonicalDocument
    from oracle_lite.db import Registry
    from talker.retrieval import CorpusRetriever

    root = tmp_path / 'output'
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    source = corpus / 'pressure.txt'
    source.write_text('pressure limit 12 kPa')
    canonical = root / '_corpora' / 'corpus-a' / 'canonical' / 'spec.json'
    CanonicalDocument(content_hash='hash-a', source_path=str(source), source_type='text',
                      parser_name='fixture', parser_version='1',
                      text=source.read_text()).write_json(canonical)
    registry = Registry(root / '_corpora' / 'corpus-a' / '_state' / 'registry.sqlite3')
    registry.upsert_source(root=corpus, path=source, size=source.stat().st_size,
                           mtime_ns=source.stat().st_mtime_ns, content_hash='hash-a')
    registry.save_artifact(content_hash='hash-a', parser_version='1',
                           canonical_path=str(canonical), status='ready')
    retriever = CorpusRetriever(root, tmp_path / 'fts.sqlite3', corpus_id='corpus-a')
    assert retriever.search('pressure limit')
    newer = canonical.with_name('spec-v2.json')
    CanonicalDocument(content_hash='hash-a', source_path=str(source), source_type='text',
                      parser_name='fixture', parser_version='2',
                      text='pressure limit 12 kPa with corrected layout').write_json(newer)
    registry.save_artifact(content_hash='hash-a', parser_version='2',
                           canonical_path=str(newer), status='ready')
    assert 'corrected layout' in retriever.search('pressure limit')[0].text
    registry.mark_missing_under_root(corpus, set())
    assert retriever.search('pressure limit') == []
    assert canonical.exists()


def test_pdf_detail_views_use_matching_small_text_and_reject_changed_source(tmp_path):
    import hashlib
    import pymupdf
    from talker.visual_evidence import pdf_detail_view
    source = tmp_path / 'spec.pdf'
    doc = pymupdf.open()
    page = doc.new_page(width=1200, height=1200)
    page.insert_text((900, 1050), 'LIMITS', fontsize=6)
    page.insert_text((900, 1070), 'Pressure 12 kPa', fontsize=8)
    doc.save(source)
    doc.close()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    image = pdf_detail_view(source, digest, 1, ['limits'], tmp_path / 'crops')
    assert image is not None and image.exists()
    assert pdf_detail_view(source, digest, 1, ['absentterm'], tmp_path / 'crops') is None
    source.write_bytes(source.read_bytes() + b'\nmodified')
    assert pdf_detail_view(source, digest, 1, ['limits'], tmp_path / 'crops') is None
