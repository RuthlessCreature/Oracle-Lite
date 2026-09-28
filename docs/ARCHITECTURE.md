# Architecture

## 1. System boundary

Oracle-Lite separates four concerns:

1. **Mutable source corpus** — files can appear, disappear, move or change.
2. **Deterministic data factory** — hashes, parses and records lineage.
3. **Immutable dataset snapshots** — fixed inputs for training/evaluation.
4. **Local model training** — consumes one explicit snapshot and writes checkpoints under the configured output root.

MiniMax M3 is outside the source-of-truth chain. It is an optional Data Janitor for low-risk enrichment only.

## 2. Three-field configuration contract

User configuration contains only:

```yaml
minimax_api_key: "..."
corpus_dir: "..."
output_dir: "..."
```

Everything else is an implementation default.

This is deliberate: training reproducibility must come from recorded run metadata and immutable snapshots, not from a growing hand-edited YAML file.

## 3. Dynamic corpus identity

The scanner uses two levels:

- `size + mtime_ns` as a fast path;
- SHA-256 as authoritative content identity.

Consequences:

- same bytes at another path => one content object;
- rename/move => path history changes, content identity does not;
- modified file => new content hash / new revision;
- deletion => tombstone, never historical erasure.

## 4. Immutable snapshot rule

A training run never reads directly from `corpus_dir`.

```text
mutable corpus
  -> registry
  -> canonical artifacts
  -> immutable snapshot manifest
  -> sharded dataset
  -> training run
```

Once created, a snapshot manifest is not edited in place.

## 5. Incremental CPT

Incremental training should not mean "train only the newest files forever".

Oracle-Lite snapshot mode supports:

```text
incremental set = new content + historical replay
```

The replay ratio is selected at snapshot creation. V0.1 defaults to 20% replay when incremental mode is used.

A later milestone can add a separate general-domain replay lane to further reduce catastrophic forgetting.

## 6. Data Janitor boundary

MiniMax M3 may produce metadata artifacts derived from source material, but its output cannot silently replace source text.

Allowed:

- document class;
- language/domain tags;
- quality flags;
- section/title recovery;
- formatting assistance;
- source-grounded paraphrase.

Forbidden as authoritative training truth:

- invented answers;
- missing-fact completion;
- changed measurements/codes/versions;
- unsourced causal explanations;
- synthetic expert reasoning presented as fact.

## 7. RTX 4080 training boundary

The built-in V0.1 profile targets one RTX 4080 16GB and favors memory safety over throughput.

Training parameters are internal code defaults and are written into each run's `run.json` so a completed run remains auditable even though the user config stays minimal.

## 8. Failure policy

A failed parser produces a recorded failed artifact instead of silently dropping a source.

A failed training run is marked `failed` in the registry and preserves existing checkpoints.

Generated output is never placed under the source corpus.
