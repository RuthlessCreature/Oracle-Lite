from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .dataset import build_cpt_dataset
from .db import Registry
from .ingest import ingest_corpus
from .scanner import scan_corpus
from .snapshot import create_snapshot
from .training import run_cpt_training

app = typer.Typer(
    name="oracle-lite",
    no_args_is_help=True,
    help="Local-first dataset factory and incremental LLM training pipeline.",
)
console = Console()


@app.command()
def init(
    config: Path = typer.Option(Path("oracle.yaml"), help="Config file to create"),
):
    """Create the minimal three-field config template."""
    if config.exists():
        raise typer.BadParameter(f"Config already exists: {config}")
    config.write_text(
        'minimax_api_key: "REPLACE_WITH_SK_CP_KEY"\n'
        'corpus_dir: "D:/OracleLite/corpus"\n'
        'output_dir: "D:/OracleLite/output"\n',
        encoding="utf-8",
    )
    console.print(
        f"[green]Created[/green] {config}. Edit only the MiniMax key, corpus path and output path."
    )


@app.command()
def scan(
    config: Path = typer.Option(Path("oracle.yaml")),
    verify_all: bool = typer.Option(False, help="Re-hash every file even if size/mtime are unchanged"),
):
    """Scan the dynamic corpus folder and update the source registry."""
    stats = scan_corpus(load_config(config), verify_all=verify_all)
    console.print_json(json.dumps(stats.as_dict()))


@app.command()
def ingest(
    config: Path = typer.Option(Path("oracle.yaml")),
    force: bool = typer.Option(False, help="Re-parse even if canonical artifact already exists"),
):
    """Parse new/changed content into canonical immutable artifacts."""
    stats = ingest_corpus(load_config(config), force=force)
    console.print_json(json.dumps(stats.as_dict()))


@app.command()
def snapshot(
    name: str = typer.Option(..., help="Human-readable snapshot name"),
    config: Path = typer.Option(Path("oracle.yaml")),
    mode: str = typer.Option("full", help="full or incremental"),
    base: str | None = typer.Option(None, help="Base snapshot id for incremental mode"),
    replay_ratio: float = typer.Option(0.20, help="Historical replay fraction for incremental snapshot"),
    seed: int = typer.Option(42),
):
    """Freeze current canonical data into an immutable training snapshot."""
    result = create_snapshot(
        load_config(config),
        name=name,
        mode=mode,
        base_snapshot_id=base,
        history_replay_ratio=replay_ratio,
        seed=seed,
    )
    console.print_json(json.dumps({
        "snapshot_id": result.snapshot_id,
        "manifest_path": str(result.manifest_path),
        "total": result.total,
        "current": result.current,
        "replay": result.replay,
    }))


@app.command("build-cpt")
def build_cpt(
    snapshot_id: str = typer.Argument(...),
    config: Path = typer.Option(Path("oracle.yaml")),
    max_records_per_shard: int = typer.Option(5000),
):
    """Build sharded CPT JSONL from a frozen snapshot."""
    result = build_cpt_dataset(
        load_config(config),
        snapshot_id=snapshot_id,
        max_records_per_shard=max_records_per_shard,
    )
    console.print_json(json.dumps({
        "output_dir": str(result.output_dir),
        "documents": result.documents,
        "characters": result.characters,
        "shards": [str(p) for p in result.shard_paths],
    }))


@app.command()
def prepare(
    name: str = typer.Option("snapshot", help="Snapshot name"),
    config: Path = typer.Option(Path("oracle.yaml")),
    mode: str = typer.Option("full", help="full or incremental"),
    base: str | None = typer.Option(None, help="Base snapshot id for incremental mode"),
    replay_ratio: float = typer.Option(0.20, help="Historical replay fraction for incremental mode"),
    verify_all: bool = typer.Option(False, help="Force SHA-256 verification for every source file"),
):
    """Run scan -> ingest -> snapshot -> CPT dataset build in one command."""
    cfg = load_config(config)
    scan_stats = scan_corpus(cfg, verify_all=verify_all)
    ingest_stats = ingest_corpus(cfg)
    snap = create_snapshot(
        cfg,
        name=name,
        mode=mode,
        base_snapshot_id=base,
        history_replay_ratio=replay_ratio,
        seed=42,
    )
    dataset = build_cpt_dataset(cfg, snapshot_id=snap.snapshot_id)

    console.print_json(json.dumps({
        "scan": scan_stats.as_dict(),
        "ingest": ingest_stats.as_dict(),
        "snapshot_id": snap.snapshot_id,
        "snapshot_total": snap.total,
        "snapshot_current": snap.current,
        "snapshot_replay": snap.replay,
        "dataset_dir": str(dataset.output_dir),
        "documents": dataset.documents,
        "characters": dataset.characters,
    }))


@app.command("train-cpt")
def train_cpt(
    snapshot_id: str = typer.Argument(..., help="Snapshot whose CPT dataset should be trained"),
    base_model: Path = typer.Option(..., exists=True, help="Local Hugging Face base-model directory"),
    config: Path = typer.Option(Path("oracle.yaml")),
    max_steps: int | None = typer.Option(None, help="Optional smoke-test cap; omit for the built-in 1 epoch preset"),
):
    """Run local RTX-4080-oriented QLoRA CPT with built-in defaults."""
    run_id = run_cpt_training(
        load_config(config),
        snapshot_id=snapshot_id,
        base_model=base_model,
        max_steps=max_steps,
    )
    console.print(f"[green]Training completed[/green] run_id={run_id}")


@app.command()
def status(config: Path = typer.Option(Path("oracle.yaml"))):
    """Show local registry summary."""
    cfg = load_config(config)
    registry = Registry(cfg.registry_path)
    with registry.connect() as conn:
        counts = {
            "active_files": conn.execute("SELECT COUNT(*) n FROM source_files WHERE status='active'").fetchone()["n"],
            "tombstoned_files": conn.execute("SELECT COUNT(*) n FROM source_files WHERE status='tombstoned'").fetchone()["n"],
            "unique_content": conn.execute("SELECT COUNT(*) n FROM content_objects").fetchone()["n"],
            "ready_artifacts": conn.execute("SELECT COUNT(*) n FROM derived_artifacts WHERE status='ready'").fetchone()["n"],
            "failed_artifacts": conn.execute("SELECT COUNT(*) n FROM derived_artifacts WHERE status='failed'").fetchone()["n"],
            "snapshots": conn.execute("SELECT COUNT(*) n FROM snapshots").fetchone()["n"],
            "training_runs": conn.execute("SELECT COUNT(*) n FROM training_runs").fetchone()["n"],
        }

    table = Table(title="Oracle-Lite Status")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in counts.items():
        table.add_row(key, str(value))
    console.print(table)


if __name__ == "__main__":
    app()
