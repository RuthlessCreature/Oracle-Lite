# Changelog

## 0.1.0 - 2026-09-28

### Added
- Dynamic single-folder corpus scanner.
- SHA-256 content identity, revisions, duplicate detection and tombstones.
- Canonical parsers for text, JSON/JSONL, PDF, DOCX and PPTX.
- Immutable full and incremental snapshots.
- Historical replay for incremental CPT snapshots.
- Sharded CPT dataset builder.
- Built-in RTX 4080 16GB QLoRA CPT preset.
- Automatic local checkpoint resume.
- MiniMax M3 Token Plan client constrained to Data Janitor tasks.
- Three-field user config: MiniMax key, corpus path, output path.
- One-command `prepare` pipeline.
- Core tests and GitHub Actions CI.

### Known limitations
- OCR for scanned/image-only PDFs is not implemented.
- Complex PPT/PDF graphics are not semantically reconstructed.
- SFT, RAG and full evaluation gates are future milestones.
