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
    corpus_dir: Path = typer.Option(Path("corpus"), help="Default corpus folder"),
):
    """Initialize a local Oracle-Lite workspace."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    if config.exists():
        raise typer.BadParameter(f"Config already exists: {config}")
    config.write_text(
        "corpus_roots:\n"
        f"  - {corpus_dir.as_posix()}\n"
        "state_dir: .oracle\n"
        "parser_version: v1\n"
        "hash_algorithm: sha256\n",
        encoding="utf-8",
    )
    cfg = load_config(config)
    Registry(cfg.registry_path)
    console.print(f"[green]Initialized[/green] config={config} corpus={corpus_dir}")


@app.command()
def scan(
    config: Path = typer.Option(Path("oracle.yaml")),
    verify_all: bool = typer.Option(False, help="Re-hash every file even if size/mtime are unchanged"),
):
    """Scan dynamic corpus folders and update the source registry."""
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


@app.command("train-cpt")
def train_cpt(
    training_config: Path = typer.Option(..., exists=True, readable=True),
    config: Path = typer.Option(Path("oracle.yaml")),
):
    """Run local QLoRA CPT from an already built immutable dataset."""
    run_id = run_cpt_training(load_config(config), training_config)
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
