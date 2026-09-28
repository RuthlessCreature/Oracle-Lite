# Changelog

## 0.3.0 - 2026-09-28

### Added
- `oracle-lite run`: one-command scan -> ingest -> snapshot -> dataset -> model download -> train/resume.
- Automatic reuse of the latest full snapshot when the active corpus hash set and parser version are unchanged.
- Up-to-date detection when the current corpus already has a completed full training run.
- Automatic checkpoint continuation when the same snapshot was interrupted or only smoke-tested.
- One-click pipeline tests covering first training, no-change idempotency, corpus mutation and smoke semantics.

### Changed
- The normal user path no longer requires manually copying a snapshot ID between `prepare` and `train`.
- One-click runs use full current-corpus snapshots rather than fragile incremental adapter chaining. Hashing still makes ingestion incremental.
- Active parser failures now stop one-click training instead of silently producing a partial corpus.
- Native image filenames are no longer treated as visual training labels.

### Configuration
The user config remains exactly three fields:
- `minimax_api_key`
- `corpus_dir`
- `output_dir`

## 0.2.0 - 2026-09-28

### Changed
- Oracle-Lite is now multimodal by default.
- Default base model is Qwen/Qwen3.5-9B-Base.
- Base model is downloaded automatically into output/models.
- Training uses AutoProcessor and AutoModelForMultimodalLM.
- User no longer supplies a base-model path.
- Canonical data now contains segment-level image assets.
- PDF pages are rasterized and paired with same-page text.
- PPTX slide images are paired with slide text.
- Native image files are accepted as corpus sources.
- RTX 4080 preset freezes base vision parameters while keeping native visual forward processing.

### Added
- Multimodal canonical segments.
- Visual asset store.
- Mixed text/visual domain dataset.
- Native image parser.
- Automatic Hugging Face model downloader.
- Multimodal parser tests.

### Safety / data quality
- Image-only records without source-grounded text are preserved but excluded from supervised visual loss.
- MiniMax remains a non-authoritative Data Janitor.

## 0.1.0 - 2026-09-28

### Added
- Dynamic corpus scanning and SHA-256 lineage.
- Immutable snapshots.
- Text-oriented domain dataset.
- MiniMax M3 Data Janitor integration.
- Three-field user configuration.
