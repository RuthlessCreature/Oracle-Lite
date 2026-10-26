# Oracle-Lite

Local-first **multimodal** dataset factory and domain-model training pipeline.

Oracle-Lite V0.6.2 is built around these fixed assumptions:

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

### Grounded retrieval and instruction examples

Talker builds a local SQLite FTS5 index from active canonical corpus segments and
adds the best matching passages to each prompt with filename and page labels.
Answers are instructed to cite those labels and to report when the retrieved
source does not support a claim. The index is stored under
`output_dir/_talker/corpus_fts.sqlite3` and updates when active source documents
change.

For instruction tuning, copy `configs/qa_pairs.example.jsonl` to
`<corpus_dir>/qa_pairs.jsonl` and add reviewed examples in JSONL format. Each
line must contain `question`, `answer`, and the exact supporting `context`; an
optional `source_path` labels the citation. The pipeline parses these as
instruction/answer records and excludes their flattened JSON from raw-text
training. Since the file is inside the corpus, edits create a new snapshot and
training dataset automatically. Snapshots freeze a hash-verified copy of the QA
file; dataset builds never add QA written after the snapshot. Long instruction
examples are rejected rather than silently dropping their evidence or answer.
Do not use model-generated answers as labels without checking the cited source.
Reviewed QA is retrieved as supporting evidence; Talker still generates the
answer and must check whether the current question actually matches it.
Instruction training and Talker share the same grounded conversation prompt,
including source attachments and the `Assistant:` answer prefix. Training
outputs include a prompt version so a new format does not automatically resume
checkpoints trained with the older instruction format. Existing adapters require
retraining to learn the shared format.

PDF text is ordered by word coordinates so visible table rows and column gaps
survive extraction. Ambiguous layouts still require inspection of the page image.
Parser version changes invalidate the old canonical parsing cache.

Use a JSONL regression suite containing `question` and `expected_evidence`
(source filename, section, optional `contains` strings). Evaluate any corpus with:

```bash
python -m oracle_lite.evaluation --output-dir /path/to/output \
  --cases /path/to/cases.jsonl --report /path/to/report.json
```

Add `--corpus-id` for outputs containing multiple corpora. Add `--generate` to
load the actual Talker adapter and save generated answers too. Optional
`answer_contains` and `answer_excludes` fields check known output regressions.
Use `answer_patterns` and `answer_forbidden_patterns` (Python regular expressions)
to check field/value pairs and reject unsupported claims. `answer_any_of` accepts
any one of several approved phrases. An `unanswerable: true` case has no
`expected_evidence` and must explicitly assert an abstention; unrelated retrieved
passages never count as correct evidence for it. Add `split` labels to report
training examples and held-out examples separately. Split labels are supplied by
the author; the evaluator does not establish that an example was held out.
Reports include the number scored as well as the number passed. Generation
without answer assertions remains unscored.
Reports distinguish retrieval recall from generation checks; string checks alone
do not establish that a generated answer is factually correct.
`configs/grounding_cases.example.jsonl` illustrates field/value pairing,
conflicting documents, and missing evidence using fictional specifications.
Replace these example sources and assertions with reviewed facts from your
corpus before evaluating a model.

For repeatable checks using unseen fictional documents, run:

```bash
python -m oracle_lite.grounding_stress --output-dir /path/to/output \
  --report-dir /path/to/reports/unseen-fields --seed 20260930 --count 16
```

This generates shuffled field/value tables, assignment lines, JSON, and Markdown
tables in English and Chinese. It tests exact field mappings, units, leading zeros,
and scientific notation using the actual loaded model. These fixtures never enter
the training corpus, and corpus retrieval and reviewed QA are disabled for this
evaluation. It measures document reading, not every hallucination mode.

To test structural generalization beyond those familiar fixture layouts, run:

```bash
python -m oracle_lite.novel_template_stress --output-dir /path/to/output \
  --report-dir /path/to/reports/unseen-document-templates --seed 20300930 --count 60
```

This uses randomized fictional values in narrative shift notes, email revisions,
records with omitted fields, prose equipment logs, nominal/tolerance descriptions,
and conflicting observations. It scores exact nested JSON, with retrieval disabled.
The task skills intentionally overlap with training while the source phrasing and
serialization structures are held out. Keep the generated cases out of training;
a report with `status: not_run` means the runtime could not access CUDA and no model
quality score was produced.

For a separate instruction-only training experiment, `oracle-lite train` accepts
`--instruction-only`, `--instruction-data`, `--gradient-accumulation-steps`,
`--learning-rate`, `--num-train-epochs`, `--base-model-path`, and
`--text-max-length`. Additional JSONL rows require
`mode: instruction`, `text` equal to `answer`, nonempty `question` and `context`,
and optionally one source image. Images are validated and frozen by their content
hash, and instruction loss covers only the complete answer and EOS. The input is
validated, frozen, and hashed into the training run;
it never becomes corpus evidence. The snapshot remains unchanged. Run metadata
records the selected training record count, input hash, and training parameters.
`--initial-adapter /path/to/adapter-final` continues an existing compatible
adapter's weights in a separate run. Its base path must match the selected local
checkpoint. The starting adapter path and weight hash are recorded; the source
adapter is never overwritten. Include retained tasks and evaluate the new run
against the old adapter before selecting it for Talker.
`--base-model-path` selects a complete local checkpoint for controlled base
model comparisons; the path and model identity are recorded with the run.
`--text-max-length` raises the no-truncation token budget for long reviewed
examples. If it is explicitly set, graceful VRAM backoff will reduce visual
work while preserving that text budget; a sample that still does not fit fails
with a clear error instead of silently dropping source or answer tokens.
`python -m oracle_lite.instruction_fixtures --output /path/to/train.jsonl` generates
explicitly fictional method-training tasks; review them and keep evaluation
seeds and source questions separate from training examples.
`python -m oracle_lite.robust_instruction_fixtures --output /path/to/train.jsonl`
generates fictional multi-scope, long row-list, identifier-disambiguation,
revision-history, and typed nested-JSON examples. Use a separate seed for
evaluation and keep those cases outside training data.

`python -m oracle_lite.visual_instruction_fixtures --output /path/to/visual.jsonl`
generates fictional source images with deterministic field/value answers for
visual instruction training. Keep these outside the corpus. Validate the source
images and combine the rows with text instructions before passing the resulting
JSONL to `--instruction-data`. For an independent seed, evaluate actual image
reading with:

```bash
python -m oracle_lite.visual_grounding_stress --output-dir /path/to/output \
  --report-dir /path/to/reports/unseen-images --seed 202609302 --count 16
```

This checks new values in four known layout families; it does not establish
accuracy on arbitrary engineering drawings. Preserve source-only PDF regression
tests and manually review field associations in addition to exact JSON scores.
Keep training and evaluation seeds separate, and report both retrieval recall
and generation behavior. Exact strings do not detect every wrong association,
invented scope, or unnecessary expansion.

Talker attaches original page images behind retrieved passages when there are no
user-supplied images, so the VLM can check spatial relationships lost in text
extraction. Source filenames and page numbers accompany the images. Dense drawing
labels still need visual review before they become instruction labels: adjacency
in extracted text does not prove that a callout belongs to a nearby identifier.


## Talker: local multimodal Web chat

Oracle-Lite V0.6.2 includes a separate `talker/` runtime for chatting with the
trained LoRA adapter without modifying training state.

Configuration:

```yaml
# talker/talker.yaml
training_output_dir: "/home/cyg/Oracle/output3Suite2"
```

`training_output_dir` may point at the Oracle-Lite output root, a training
directory, or an adapter directory. Talker recursively discovers the newest
usable `adapter-final` or checkpoint, reads its `run.json` when available,
and resolves the matching local Base model automatically.

Install Talker dependencies:

```bash
python -m pip install -U -e ".[talk]"
```

One-click local launch:

```bash
oracle-talker
```

Equivalent forms:

```bash
python -m talker
bash talker/start.sh
```

Talker starts at port 7870 (or the next free port), listens on all local
interfaces, and prints all usable access URLs. It shows both
`http://127.0.0.1:<port>/` and detected private-LAN addresses such as
`http://192.168.x.x:<port>/`. The browser prefers the LAN URL when available,
which is useful when a VPN/system proxy intercepts localhost. If the VPN has a
"Block LAN" mode, enable its local-network/private-address bypass. Talker has no
authentication, so do not expose this port to the public Internet.

Talker provides:

- Base + LoRA 4-bit multimodal inference;
- persistent SQLite conversation history;
- new / switch / delete conversations;
- text chat and multi-file upload;
- direct image vision input;
- bounded PDF text + page-image extraction;
- DOCX / PPTX bounded XML/media extraction;
- TXT / Markdown / CSV / JSON / JSONL context;
- STEP / STP / Parasolid X_T / X_B mechanical-CAD context;
- latest-Adapter reload from the Web UI;
- a 512 MiB per-file upload ceiling;
- recent-image prioritization and bounded conversation context;
- automatic long-answer continuation in bounded generation chunks;
- no fixed total answer-token ceiling: Talker continues until EOS/completion;
- only a rolling continuation tail is re-fed to the model, so very long answers do not make each inference prompt grow without bound;
- repeated/non-growing continuation detection prevents runaway loops without imposing a length limit;
- CUDA OOM retry with reduced context/image count.

Talker stores only its own local state under `talker/state/`; it **reads** the
training output but never writes into the adapter/training directories.

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

## Oracle Viewer

本地三维特征与海克斯康测量方法查看入口：`/viewer`，与 TALKER 共用模型服务和项目状态。支持 MSOP 索引/原页/依赖链、STEP 逐面显示与人工 FAI 定位、项目隔离的方案草案任务。原项目 X_T 已本地解码并显示真实几何预览，支持装配实例、金属材质及旋转/缩放/剖切；严格 B-Rep/计量转换仍待验证，FAI 绑定暂未启用。

实现与验证边界见 [Oracle Viewer 文档](oracle_viewer/README.md)。

FAI 标记的 MSOP 符号、候选面约束投影和逐项复核流程见 [FAI 标记迭代文档](docs/fai-overlay-refinement.md)。

### 通用 AOI 方案工作流与规格库

Viewer 项目工具栏的 **方案工作流 · LIBRARY** 提供完整六步页面。本地 Viewer 模型作出候选选型/FAI 合并决策，工程工具执行光学、规格和覆盖校核，场景与 BOM 使用同一配置。首批厂商原始规格已落盘，613-13599 Rev06 的 FAI23/24 已完成实际 Strata 决策试点。方法/位置与计量验证缺口仍显式保留。

详见 [通用 AOI 工作流与试点结果](docs/aoi-workflow.md)。
