from oracle_lite.canonical import CanonicalDocument, CanonicalSegment
from talker.engineering import retrieval_query
from talker.retrieval import CorpusRetriever


def test_design_followup_keeps_feature_but_explicit_replacement_wins():
    assert retrieval_query(['针对FAI23出视觉方案', '相机光源和BOM呢？']).startswith('FAI 23 MSOP')
    assert retrieval_query(['FAI23方案', 'FAI24方案']) == 'FAI 24 方案'
    assert retrieval_query(['MSOP里面的测量项', '所有FAI项汇总']) == 'MSOP 所有FAI项汇总'
    assert retrieval_query(['Revision B pressure?', 'Revision C?']) == 'Revision C?'


def test_inventory_scans_beyond_top_three_and_preserves_conflicting_note(tmp_path):
    canonical = tmp_path / '_corpora' / 'corpus-a' / 'canonical' / 'msop.json'
    segments = [CanonicalSegment(
        text=f'Part | MSOP: FAI {n} nominal {n}.00 +0.12/-0.12',
        metadata={'page': n},
    ) for n in range(1, 9)]
    segments.append(CanonicalSegment(
        text='Part | MSOP: FAI 54 to FAI 60\nFAI61-FAI79 are weld spots height, can’t measure by 443DZ',
        metadata={'page': 103},
    ))
    CanonicalDocument(content_hash='msop-hash', source_path='/source/example MSOP.pdf',
                      source_type='pdf', parser_name='test', parser_version='1', text='',
                      segments=segments).write_json(canonical)
    retriever = CorpusRetriever(tmp_path, tmp_path / 'index.sqlite3')
    context = retriever.context('MSOP 所有FAI项目', max_chars=8000)
    assert '9 of 9' in context and 'page 8' in context
    assert 'FAI 54 to FAI 60' in context and 'FAI61-FAI79' in context
    limited = retriever.context('MSOP 所有FAI项目', max_chars=650)
    assert len(limited) <= 650
    assert 'of 9' in limited


def test_design_mode_without_retrieval_still_generates_and_does_not_mutate_history(tmp_path, monkeypatch):
    from talker.model import TalkerModel
    model = TalkerModel(tmp_path)
    model.model = object()
    model.processor = object()

    class EmptyRetriever:
        def curated_context(self, *args, **kwargs):
            return None

        def context(self, *args, **kwargs):
            return ''

    model.retriever = EmptyRetriever()

    def generate(messages, **kwargs):
        if len(messages) == 1:
            assert 'preliminary optical inspection proposal' in messages[-1]['content']
            assert 'not measurement accuracy' in messages[-1]['content']
        else:
            assert messages[-2]['content'] == '候选BOM与设计假设'
            assert 'ORIGINAL attachments' in messages[-1]['content']
        return '候选BOM与设计假设'

    monkeypatch.setattr(model, '_answer_with_auto_continue', generate)
    messages = [{'role': 'user', 'content': '出一个光学方案和BOM'}]
    assert model.answer(messages) == '候选BOM与设计假设'
    assert messages[0]['content'] == '出一个光学方案和BOM'


def test_alignment_dependencies_keep_program_scope_and_final_origin(tmp_path):
    from talker.retrieval import RetrievedPassage
    canonical = tmp_path / '_corpora' / 'corpus-a' / 'canonical' / 'msop.json'
    texts = [
        'Part | MSOP: FAI 23\nMeasured System ALI_ABD\nProgram UP',
        'Part | MSOP: ALI_ABD\nMeasured System ALI_ABC\nDATUMB_LIN and DATUMD_LIN intersection BD_ORIGN sets Y zero\nProgram UP',
        'Part | MSOP: ALI_ABC\nDATUMA_PLN levels Z\nProgram UP',
        'Part | MSOP: DATUM D\nConstruct middle line DATUMD_LIN\nProgram UP',
        'Part | MSOP: ALI_ABD\nWRONG_DOWN_ORIGIN\nProgram DOWN',
    ]
    CanonicalDocument(content_hash='msop-hash', source_path='/source/example MSOP.pdf',
                      source_type='pdf', parser_name='test', parser_version='1', text='',
                      segments=[CanonicalSegment(text=text, metadata={'page': n})
                                for n, text in enumerate(texts, 1)]).write_json(canonical)
    retriever = CorpusRetriever(tmp_path, tmp_path / 'index.sqlite3')
    context = retriever.alignment_context(
        RetrievedPassage('example MSOP.pdf', 'page 1', texts[0], 'msop-hash'), max_chars=4000,
    )
    assert 'BD_ORIGN' in context and 'Construct middle line' in context
    assert 'WRONG_DOWN_ORIGIN' not in context
