# Oracle-Lite

Local-first **multimodal** dataset factory and domain-model training pipeline.

Oracle-Lite V0.2 is built around these fixed assumptions:

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

## First run

Create config:

```bash
oracle-lite init
```

Put source files into `corpus_dir`, then:

```bash
oracle-lite prepare --name initial
```

This executes:

```text
scan
-> multimodal ingest
-> immutable snapshot
-> mixed domain dataset
```

It prints a `snapshot_id`.

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

## Train

Smoke test:

```bash
oracle-lite train <snapshot_id> --max-steps 10
```

Full built-in run:

```bash
oracle-lite train <snapshot_id>
```

V0.2 RTX 4080 policy:

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

## Incremental update

After the corpus changes:

```bash
oracle-lite prepare \
  --name update-001 \
  --mode incremental \
  --base <previous_snapshot_id>
```

Hash identity handles duplicates and renames. New content is combined with deterministic historical replay. A running training job never reads the live corpus directly.

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

## Known V0.2 limits

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
