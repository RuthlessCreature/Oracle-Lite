# Oracle-Lite V0.2 Architecture

## 1. Hard requirements

Oracle-Lite is multimodal by default.

The system must preserve both semantic text and source visuals from heterogeneous documents. A VLM checkpoint loaded with text-only datasets does not satisfy this requirement.

Default base model:

```text
Qwen/Qwen3.5-9B-Base
```

## 2. Three-field configuration contract

```yaml
minimax_api_key: "..."
corpus_dir: "..."
output_dir: "..."
```

No base-model path is configured. Oracle-Lite owns model selection and download.

## 3. Source layer

Supported source classes:

- TXT / Markdown / CSV;
- JSON / JSONL;
- PDF;
- DOCX;
- PPTX;
- PNG / JPG / JPEG / WebP / BMP / TIFF.

The source directory is mutable.

## 4. Identity and lineage

```text
size + mtime
   -> fast unchanged check
SHA-256
   -> authoritative content identity
```

Rename/move does not duplicate content. A changed file becomes a new revision. Deletion creates a tombstone.

## 5. Multimodal Canonical Layer

A canonical document contains:

```text
document
├── content_hash
├── source_path
├── text
├── metadata
└── segments[]
    ├── text
    ├── images[]
    └── metadata
```

The important unit is the segment:

- PDF => one segment per page;
- PPTX => one segment per slide;
- DOCX => document-level segment with extracted media in V0.2;
- native image => visual segment.

## 6. Visual asset policy

Visual assets are copied/rendered under:

```text
output_dir/assets/<hash-prefix>/<content-hash>/
```

Generated assets never live under `corpus_dir`.

PDF pages are rasterized at 1.5x. Office vector formats that common image loaders cannot safely decode are excluded in V0.2 rather than silently fed into the processor.

## 7. Dataset objective

The domain dataset mixes two objectives.

### Text lane

Source text is used for ordinary causal language modeling.

### Visual lane

Only records with both:

- source visual asset;
- source-grounded text;

enter visual supervision.

The deterministic target is text extracted from that same source page/slide/document. MiniMax does not invent the answer.

Unlabeled visual records are retained but filtered out of the current training loss.

## 8. Model lifecycle

```text
Qwen/Qwen3.5-9B-Base
        |
        v
output/models/Qwen3.5-9B-Base
        |
        v
4-bit load
        |
        v
native AutoProcessor + AutoModelForMultimodalLM
        |
        v
QLoRA domain adaptation
```

The model is downloaded once through Hugging Face Hub and reused locally.

## 9. RTX 4080 policy

One RTX 4080 16GB is the design envelope.

Therefore V0.2 uses:

- 4-bit NF4;
- BF16 compute;
- batch size 1;
- gradient checkpointing;
- LoRA rank 8;
- long gradient accumulation;
- processor-native multimodal image handling;
- frozen base vision parameters.

Freezing the vision tower means its original weights do not receive optimizer state. It does **not** mean images are removed from the forward pass.

## 10. Snapshot boundary

Training never reads directly from the mutable corpus.

```text
corpus
-> registry
-> canonical text/assets
-> immutable snapshot
-> mixed domain shards
-> training run
```

A corpus change during a run affects only future snapshots.

## 11. MiniMax boundary

MiniMax M3 is a Data Janitor only.

Its output may enrich metadata, but it cannot silently replace source truth used as a training target.

## 12. Failure policy

- parser failures are recorded;
- unsupported visual assets are skipped explicitly;
- image-only samples without reliable text are not fabricated into supervised examples;
- failed training runs retain checkpoints;
- every completed run records base model ID, snapshot ID and preset values.
