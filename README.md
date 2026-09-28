# Oracle-Lite

Local-first **multimodal** dataset factory and domain-model training pipeline.

Oracle-Lite V0.5.0 is built around these fixed assumptions:

- corpus files live in one continuously changing local folder;
- text, images, PDF pages, Word media and PowerPoint media are first-class source material;
- the default base model is **Qwen/Qwen3.5-9B-Base**;
- the base model is downloaded automatically into `output_dir/models/`;
- local training targets one RTX 4080 16GB and prioritizes memory safety over speed;
- MiniMax M3 Token Plan is a low-risk **Data Janitor**, never the source of domain truth.

## User configuration

The editable config intentionally has exactly three fields:

```yaml
minimax_api_key: "sk-cp-REPLACE_ME"
corpus_dir: "D:/OracleLite/corpus"
output_dir: "D:/OracleLite/output"
```

Copy `configs/oracle.example.yaml` to `oracle.yaml`. The file is gitignored.

All parser settings, model ID, model download path, hashing, snapshot layout and RTX 4080 training defaults are internal code defaults.

### Hash cache and restart behavior

Oracle-Lite always walks the corpus directory on startup so it can discover new, deleted or renamed files, but it does **not** recompute SHA-256 for unchanged files. If path, size and nanosecond mtime match the registry, the existing content hash is reused immediately.

V0.5.0 gives every canonical `corpus_dir` path a stable `corpus_id`. Registry, Canonical assets, snapshots, datasets, training outputs and logs live under `output_dir/_corpora/<corpus_id>/`. Switching `corpus_dir` therefore cannot mix active files, snapshots or training state. Switching back returns to that corpus's own hash cache. A v0.4 single-corpus registry is migrated automatically only when every recorded root exactly matches the current `corpus_dir`; ambiguous/mixed legacy state is never imported.

## What is multimodal here?

Oracle-Lite does not merely load a VLM and then throw the images away.

```text
dynamic corpus
  |
  +-- TXT / MD / CSV / JSON / JSONL
  +-- PDF
  +-- DOCX
  +-- PPTX
  +-- PNG / JPG / JPEG / WEBP / BMP / TIFF
  +-- STEP / STP / Parasolid X_T / X_B mechanical CAD
  |
  v
SHA-256 Registry
  |
  v
Multimodal Canonical Layer
  |
  +-- text
  +-- image assets
  +-- page / slide / segment relationship
  |
  v
Immutable Snapshot
  |
  v
Mixed Domain Dataset
  |
  +-- text-only CLM records
  +-- image + source-text grounded records
  |
  v
Qwen3.5-9B-Base 4-bit QLoRA
```

### PDF

Each PDF page is preserved as:

- page raster image;
- text extracted from that same page;
- page number metadata.

This creates deterministic image/text grounding without a teacher model.

### PPTX

Each slide preserves:

- slide text and tables in layout order;
- raster images embedded in that slide;
- slide number metadata.

### DOCX

Word text/tables are retained and raster media inside the document are extracted as visual assets.

### Mechanical 3D CAD

`.stp` / `.step` and Parasolid `.x_t` / `.x_b` are first-class corpus sources.

- the original CAD file is preserved losslessly as a Canonical asset;
- STEP is scanned in bounded streaming mode for schema and deterministic entity/type statistics;
- X_T is scanned conservatively for published transmit/schema metadata and coarse node tokens;
- X_B is preserved as an opaque binary geometry asset rather than guessed into text;
- these metadata summaries are **not** treated as authoritative geometric labels;
- exact B-Rep/topology/tessellation/render supervision requires a CAD kernel stage and remains isolated from the core parser.

### Native images

Image files are retained as visual canonical assets. If there is no source-grounded text label, they are preserved but not used as supervised visual training targets yet.

## Why unlabeled images are not invented into training data

With no strong teacher model, Oracle-Lite does **not** fabricate captions or expert answers for image-only material.

A visual record is trained only when reliable source-grounded text exists, such as:

- PDF page image + PDF text layer;
- PPT slide images + slide text;
- DOCX media + document text.

Scanned PDFs and image-only sources remain available in Canonical storage for later OCR/RAG work.

## MiniMax M3

MiniMax is configured through the China endpoint in `oracle_lite/minimax.py`.

Allowed Data Janitor work:

- classification;
- metadata extraction;
- title/section recovery;
- quality flags;
- formatting cleanup;
- source-grounded paraphrase.

Forbidden as authoritative training truth:

- inventing professional answers;
- filling missing facts;
- changing numbers, units, alarm codes, dates or versions;
- generating unsupported reasoning.

## Install

Data factory:

```bash
pip install -e .
```

Training environment:

```bash
pip install -e ".[train]"
```

## One-click run

Create config:

```bash
oracle-lite init
```

Put source files into `corpus_dir`, then run exactly:

```bash
oracle-lite run
```

The command automatically starts a local Web Training Console and opens it in the default browser. It prefers `127.0.0.1:7860` and automatically falls back to another localhost port if 7860 is occupied. No dashboard setting is added to `oracle.yaml`.

That single command performs:

```text
read 3-field config
-> scan/hash corpus
-> parse only new/changed content
-> pause/retry on memory pressure; fail only on genuine parser/content errors
-> freeze/reuse a full current-corpus snapshot
-> build/reuse the multimodal domain dataset
-> auto-download Qwen/Qwen3.5-9B-Base if missing
-> train or resume from checkpoint
```

If the corpus has not changed and the current snapshot already completed a full training run, `oracle-lite run` returns `up_to_date` without retraining.

For a short local GPU smoke test:

```bash
oracle-lite run --max-steps 10
```

A smoke run never marks the snapshot as fully trained.

## Base model download

You do not configure a model path.

Optional pre-download:

```bash
oracle-lite download-model
```

Default model:

```text
Qwen/Qwen3.5-9B-Base
```

Local cache:

```text
output_dir/models/Qwen3.5-9B-Base/
```

If the model is absent when training starts, Oracle-Lite downloads it automatically.

## Web Training Console

The console starts before corpus scanning and remains active through model save.

It displays:

- pipeline stage: scan / ingest / snapshot / dataset / model download / model load / train / save;
- global step, total steps, progress percentage, epoch, loss, learning rate, grad norm, elapsed time and ETA;
- current checkpoint and resume status;
- GPU utilization, temperature, power and VRAM used / total;
- CPU utilization;
- system RAM, available RAM, reserved RAM, swap pressure, Oracle-Lite RSS and whole process-tree RSS;
- disk used / free / total;
- corpus counts: seen / new / changed / duplicates / tombstones / parse failures;
- dataset counts: total / text / visual records;
- live application and Trainer logs.

The server binds to localhost only. Runtime logs are also persisted under `output_dir/logs/` as JSONL, together with a final JSON snapshot of the dashboard state.

If NVML/GPU telemetry is unavailable, GPU monitoring degrades gracefully and training continues.

### Hard resource redlines: delay, never resource-kill

Oracle-Lite V0.4.4 uses **admission control**. Resource pressure is handled before the next heavy operation starts.

**RAM**
- On a ~64 GiB Ubuntu host, Oracle-Lite keeps roughly 19–20 GiB as an OS reserve.
- A parser starts only after available RAM recovers to roughly 26 GiB.
- Each parser worker receives a Linux `RLIMIT_AS` ceiling of roughly 6 GiB.
- Oracle-Lite does not call `terminate()`, `kill()`, `SIGKILL` or `os.kill()` for resource pressure.
- If an allocation is denied and the worker reports `MemoryError`, the file is retried with a lower-footprint streaming parser; it is not marked failed because of memory pressure.

**Disk**
- A large fixed/percentage reserve is kept on the filesystem containing `output_dir`.
- Corpus scan temp indexes, Canonical assets, PDF page images, Dataset shards, model download, checkpoints and final adapter saves all check disk headroom before writing.
- If free space is below the resume threshold, the pipeline stays in `WAITING_FOR_DISK` until space is available.
- Console logs rotate at 50 MiB with three retained rotations.
- Snapshot SQLite writes commit/checkpoint in bounded batches so the WAL cannot grow without bound.

**VRAM**
- Model placement is capped to about 12 GiB on the RTX 4080.
- Each training step requires about 2.5 GiB free VRAM before it starts.
- Supervised visual records contain at most one image.
- Qwen image processing is capped to a bounded pixel area; PDF page raster images are also created under an explicit pixel budget.
- CUDA OOM is caught: Oracle-Lite clears cache, lowers visual/text footprint, waits for VRAM, and retries from the latest checkpoint instead of exiting.

**Corpus-scale memory**
- scan seen-path state is stored in a temporary SQLite index instead of a Python `set`;
- active source iteration uses SQLite cursors instead of `fetchall()`;
- full/incremental snapshot creation streams rows and uses bounded batches;
- SQLite temp work is forced to disk with a bounded page cache;
- Dataset training uses local JSONL streaming instead of Arrow materialization.

Hash reuse remains unchanged: if path + size + nanosecond mtime match the registry, Oracle-Lite reuses the existing SHA-256 without reading the full file again.

## Advanced/manual commands

Normal use should be `oracle-lite run`.

The lower-level commands remain available for debugging or controlled experiments:

```bash
oracle-lite scan
oracle-lite ingest
oracle-lite prepare --name manual
oracle-lite build-domain <snapshot_id>
oracle-lite train <snapshot_id>
oracle-lite download-model
```

V0.4.4 RTX 4080 policy:

- Qwen3.5-9B-Base;
- 4-bit NF4;
- BF16 compute;
- LoRA rank 8;
- micro-batch 1;
- gradient accumulation 32;
- gradient checkpointing;
- paged 8-bit AdamW;
- native vision tower participates in forward processing;
- vision tower base parameters are frozen;
- trainable LoRA is kept outside visual/vision modules by default;
- automatic checkpoint resume.

The vision tower is frozen for memory safety, not bypassed. Image inputs still traverse the native Qwen3.5 visual encoder.

## Dynamic corpus updates

After adding, modifying, deleting, moving or duplicating source files, simply run:

```bash
oracle-lite run
```

Hash identity prevents duplicate parsing and treats renames/moves as the same content. The one-click path intentionally creates a **full current-corpus snapshot** whenever the active content set changes, then trains from the fixed Base model. This is slower than adapter chaining but materially safer and more reproducible on a single 4080. A running training job never reads the live corpus directly.

## Output layout

```text
output_dir/
├── _corpora/
│   └── corpus-<stable-id>/
│       ├── _state/registry.sqlite3
│       ├── assets/
│       ├── canonical/
│       ├── snapshots/
│       ├── datasets/
│       ├── training/
│       └── logs/
└── models/
    └── Qwen3.5-9B-Base/
```

## Known V0.5.0 limits

- scanned PDF pages are visually preserved but OCR is not yet used as a deterministic label source;
- PPTX embedded raster images are preserved, but the entire slide is not rendered into one screenshot;
- DOCX media-to-paragraph anchoring is coarse;
- vision tower base weights are frozen on the 4080 preset;
- exact CAD B-Rep/topology/tessellation/render supervision is not yet generated; v0.5.0 preserves CAD truth and deterministic metadata without inventing geometry labels;
- true large-scale multimodal continued pretraining is outside a single 16GB GPU envelope.

## Development

```bash
pip install -e ".[dev]"
pytest -q
python -m compileall -q oracle_lite
```

See `docs/ARCHITECTURE.md`.
