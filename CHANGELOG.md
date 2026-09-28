# Changelog

## 0.4.2 - 2026-09-28

### Fixed
- Corpus parsing now runs one source file per isolated worker process.
- Parser-worker RSS and system available RAM are monitored continuously; unsafe workers are terminated before exhausting the workstation.
- Linux parser workers receive an OS-level `RLIMIT_AS` hard cap before opening source content.
- `oracle-lite run` and `oracle-lite train` keep at least 25% / 12 GiB host RAM reserved and terminate immediately if RAM crosses that redline.
- Swap growth above the startup baseline is capped at roughly 512 MiB; exceeding it triggers an emergency stop before swap thrashing can freeze the desktop.
- Canonical serialization no longer deep-copies the full document or constructs one giant JSON string.
- Canonical segments are written to streamable JSONL sidecars and Dataset construction streams them line-by-line.
- PDF parsing releases page/pixmap objects incrementally and avoids duplicate full-document text aggregation.
- DOCX media extraction uses bounded streaming copy.
- JSONL is read line-by-line; JSON/text outputs are chunked instead of building additional whole-document concatenations.
- Visual assets are isolated by parser version so V3 rebuilding cannot invalidate old snapshots.

### Hash/cache behavior
- Existing SHA-256 values are preserved across this upgrade.
- Unchanged files reuse registry hashes when path + size + mtime_ns match; the second scan performs zero content hashing.
- Parser version is now `v3-memory-safe`, so existing content is reparsed once into the new Canonical layout, then reused on later runs.

### Configuration
The user config remains exactly three fields:
- `minimax_api_key`
- `corpus_dir`
- `output_dir`

## 0.4.1 - 2026-09-28

### Fixed
- Corpus scanning now streams live progress to the Web Training Console instead of updating only after the full scan completes.
- Large-file SHA-256 hashing reports processed bytes and percentage while the file is being hashed.
- The dashboard shows the current file being scanned and live hash progress, eliminating the appearance of a frozen first-run scan.

## 0.4.0 - 2026-09-28

### Added
- Auto-opened local Web Training Console for both `oracle-lite run` and manual `oracle-lite train`.
- Live pipeline stages from corpus scan through final adapter save.
- Hugging Face Trainer telemetry: step, max steps, progress, epoch, loss, learning rate, grad norm, elapsed time and ETA.
- Checkpoint save/resume visibility.
- GPU telemetry through NVML: utilization, VRAM, temperature and power.
- CPU, system RAM, Oracle-Lite process RSS and disk telemetry.
- Live in-browser logs plus persistent JSONL logs.
- Final run-state JSON under `output_dir/logs/`.
- Automatic localhost port fallback when 7860 is occupied.

### Configuration
No configuration keys were added. User configuration remains exactly:
- `minimax_api_key`
- `corpus_dir`
- `output_dir`

### Failure behavior
- Web/system telemetry failures do not abort training.
- Training/data failures are displayed in the Web console and persisted in the final state report.

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
