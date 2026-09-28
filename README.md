# Oracle-Lite

Local-first **multimodal** dataset factory and domain-model training pipeline.

Oracle-Lite V0.4.2 is built around these fixed assumptions:

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

After upgrading to V0.4.2, existing hashes are still reused. The parser version changes to `v3-memory-safe`, so existing sources are parsed once into the new memory-safe Canonical format; subsequent runs reuse those Canonical artifacts too.

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

### Memory safety during corpus parsing

Each source file is parsed in an isolated worker process. On Linux, the worker receives an OS-level `RLIMIT_AS` hard address-space cap before it opens source content. Oracle-Lite also keeps a conservative host-RAM reserve (at least 25% / 12 GiB, whichever is larger), monitors parser-worker RSS, and allows only about 512 MiB of additional swap growth during a run. If RAM pressure crosses a redline, Oracle-Lite releases the current parser worker, enters `WAITING_FOR_MEMORY`, waits for RAM to recover above a higher resume threshold, and retries the same file. Memory pressure is not recorded as a failed artifact. Parser workers still keep a Linux `RLIMIT_AS` hard ceiling so one file cannot allocate the whole workstation.

Canonical segments are stored in streamable JSONL sidecars, and Dataset construction reads both snapshot manifests and Canonical segments line-by-line. PDF page objects are released as they are rasterized; DOCX media extraction and text/JSONL readers use bounded streaming where possible. If a file repeatedly triggers pressure, Oracle-Lite switches to a lower-memory fallback: raw text/JSON streaming, incremental Office XML extraction, or progressively lower PDF raster scale. Multi-image source segments are expanded to one image per training record.

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

V0.4.2 RTX 4080 policy:

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
├── _state/
│   └── registry.sqlite3
├── assets/
├── canonical/
├── snapshots/
├── datasets/
├── models/
│   └── Qwen3.5-9B-Base/
├── training/
└── logs/
```

## Known V0.4.2 limits

- scanned PDF pages are visually preserved but OCR is not yet used as a deterministic label source;
- PPTX embedded raster images are preserved, but the entire slide is not rendered into one screenshot;
- DOCX media-to-paragraph anchoring is coarse;
- vision tower base weights are frozen on the 4080 preset;
- true large-scale multimodal continued pretraining is outside a single 16GB GPU envelope.

## Development

```bash
pip install -e ".[dev]"
pytest -q
python -m compileall -q oracle_lite
```

See `docs/ARCHITECTURE.md`.
