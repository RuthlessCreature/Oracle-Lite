# Changelog

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
