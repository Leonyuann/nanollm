from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(slots=True)
class RunArtifacts:
    """Filesystem layout for one training run.

    Extended description.

    Attributes:
        run_dir: Root directory for the run.
        checkpoints_dir: Directory containing saved checkpoints.
        tokenizer_dir: Directory containing copied tokenizer artifacts.
        eval_dir: Directory containing evaluation outputs.
        samples_dir: Directory containing generation samples.
        metrics_path: JSONL metrics log path.
        summary_path: JSON summary path shared by scripts.
        config_snapshot_path: Saved config snapshot path.
    """

    run_dir: Path
    checkpoints_dir: Path
    tokenizer_dir: Path
    eval_dir: Path
    samples_dir: Path
    metrics_path: Path
    summary_path: Path
    config_snapshot_path: Path


def build_run_artifacts(runs_root: str | Path, run_name: str) -> RunArtifacts:
    """Create the directory layout for a run.

    Args:
        runs_root: Root directory containing all runs.
        run_name: Unique name for the new run.

    Returns:
        The created run-artifact path bundle.

    Raises:
        FileExistsError: If the target run directory already exists.
        ValueError: If ``run_name`` is empty.
    """
    if not run_name:
        raise ValueError("run_name must not be empty")

    run_dir = Path(runs_root) / run_name
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    checkpoints_dir = run_dir / "checkpoints"
    tokenizer_dir = run_dir / "tokenizer"
    eval_dir = run_dir / "eval"
    samples_dir = run_dir / "samples"
    for directory in (
        checkpoints_dir,
        tokenizer_dir,
        eval_dir,
        samples_dir,
    ):
        directory.mkdir(parents=True, exist_ok=False)

    return RunArtifacts(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        tokenizer_dir=tokenizer_dir,
        eval_dir=eval_dir,
        samples_dir=samples_dir,
        metrics_path=run_dir / "metrics.jsonl",
        summary_path=run_dir / "summary.json",
        config_snapshot_path=run_dir / "config.snapshot.yaml",
    )


def append_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file.

    Args:
        path: Destination JSONL path.
        record: Serializable record to append.

    Returns:
        None.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
        output_file.write("\n")


def copy_tokenizer_artifacts(
    vocab_path: str | Path,
    merge_path: str | Path,
    destination_dir: str | Path,
) -> tuple[Path, Path]:
    """Copy tokenizer artifacts into a run directory.

    Args:
        vocab_path: Source vocabulary artifact path.
        merge_path: Source merge artifact path.
        destination_dir: Destination tokenizer directory.

    Returns:
        A tuple ``(copied_vocab_path, copied_merge_path)``.
    """
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    copied_vocab_path = destination / "vocab.txt"
    copied_merge_path = destination / "merges.txt"
    shutil.copy2(vocab_path, copied_vocab_path)
    shutil.copy2(merge_path, copied_merge_path)
    return copied_vocab_path, copied_merge_path


def deep_merge_dicts(
    base: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge nested dictionaries.

    Args:
        base: Existing dictionary state.
        updates: New dictionary values that override or extend ``base``.

    Returns:
        The merged dictionary.
    """
    merged = dict(base)
    for key, value in updates.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def update_summary(path: str | Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge updates into a run summary JSON file.

    Args:
        path: Summary JSON path.
        updates: Partial summary payload to merge.

    Returns:
        The full merged summary payload.
    """
    summary_path = Path(path)
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as summary_file:
            current_summary = json.load(summary_file)
    else:
        current_summary = {}

    merged_summary = deep_merge_dicts(current_summary, updates)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(merged_summary, summary_file, indent=2, ensure_ascii=True, sort_keys=True)
        summary_file.write("\n")
    return merged_summary


def write_config_snapshot(
    source_path: str | Path,
    destination_path: str | Path,
) -> None:
    """Copy the active YAML config into a run directory.

    Args:
        source_path: Source config path.
        destination_path: Destination snapshot path.

    Returns:
        None.
    """
    snapshot_path = Path(destination_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, snapshot_path)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a checkpoint payload with ``torch.save``.

    Args:
        path: Destination checkpoint path.
        payload: Serializable checkpoint payload.

    Returns:
        None.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def resolve_run_dir(
    run_dir: str | None,
    checkpoint: str,
) -> Path:
    """Resolve the run directory from CLI inputs.

    Args:
        run_dir: Explicit run-directory argument, if provided.
        checkpoint: Checkpoint selector or explicit checkpoint path.

    Returns:
        The resolved run directory path.

    Raises:
        ValueError: If a named checkpoint is used without a run directory.
    """
    if run_dir is not None:
        return Path(run_dir)
    if checkpoint in {"best", "latest"}:
        raise ValueError("run_dir is required when checkpoint is 'best' or 'latest'")
    return Path(checkpoint).resolve().parents[1]


def resolve_checkpoint_path(
    run_dir: str | Path | None,
    checkpoint: str,
) -> Path:
    """Resolve a checkpoint selector into a concrete path.

    Args:
        run_dir: Run directory containing the checkpoints directory.
        checkpoint: ``best``, ``latest``, or an explicit checkpoint path.

    Returns:
        The concrete checkpoint path.

    Raises:
        ValueError: If a named checkpoint is used without a run directory.
    """
    if checkpoint in {"best", "latest"}:
        if run_dir is None:
            raise ValueError("run_dir is required when checkpoint is 'best' or 'latest'")
        return Path(run_dir) / "checkpoints" / f"{checkpoint}.pt"
    return Path(checkpoint)
