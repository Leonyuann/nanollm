from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch

import config_manager


def _load_script(script_name: str) -> ModuleType:
    """Load a repository script as an importable module.

    Args:
        script_name: Script filename under ``scripts/``.

    Returns:
        The loaded script module.
    """
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".", "_"), script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_tokenizer_files(tmp_path: Path) -> tuple[Path, Path]:
    """Write a byte-level tokenizer artifact pair for script tests.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        A ``(vocab_path, merge_path)`` tuple.
    """
    vocab_path = tmp_path / "vocab.txt"
    merge_path = tmp_path / "merges.txt"
    vocab_lines = [f"{idx}\t{idx:02x}" for idx in range(256)]
    vocab_path.write_text("\n".join(vocab_lines) + "\n", encoding="utf-8")
    merge_path.write_text("", encoding="utf-8")
    return vocab_path, merge_path


def _write_config(
    tmp_path: Path,
    vocab_path: Path,
    merge_path: Path,
    runs_root: Path,
    train_path: Path,
    valid_path: Path,
) -> Path:
    """Write a small workflow config for integration tests.

    Args:
        tmp_path: Temporary directory fixture.
        vocab_path: Tokenizer vocabulary path.
        merge_path: Tokenizer merge path.
        runs_root: Run-artifact root directory.
        train_path: Training corpus path.
        valid_path: Validation corpus path.

    Returns:
        The config file path.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "bpe:",
                "  num_process: 1",
                "  vocab_size: 256",
                "  special_tokens: []",
                f'  vocab_path: "{vocab_path}"',
                f'  merge_path: "{merge_path}"',
                "data:",
                f'  owt_train_path: "{train_path}"',
                f'  owt_valid_path: "{valid_path}"',
                f'  TinyStories_train_path: "{train_path}"',
                f'  TinyStories_valid_path: "{valid_path}"',
                "artifacts:",
                f'  runs_root: "{runs_root}"',
                "model:",
                "  vocab_size: 256",
                "  max_seq_len: 4",
                "  d_model: 16",
                "  num_layers: 2",
                "  num_heads: 4",
                "  ffn_hidden_dim: 32",
                '  norm_type: "rmsnorm"',
                '  ffn_type: "swiglu"',
                "  use_residual: true",
                "  dropout: 0.0",
                "  tie_embeddings: false",
                "  rope_base: 10000.0",
                "  bias: true",
                "training:",
                '  dataset: "tinystories"',
                "  batch_size: 2",
                "  max_steps: 2",
                "  learning_rate: 1.0e-3",
                "  warmup_steps: 1",
                "  log_interval: 1",
                "  eval_interval: 1",
                "  eval_steps: 1",
                '  device: "cpu"',
                "  seed: 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _prepare_small_corpora(tmp_path: Path) -> tuple[Path, Path]:
    """Write tiny train and validation corpora.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        A ``(train_path, valid_path)`` tuple.
    """
    train_path = tmp_path / "train.txt"
    valid_path = tmp_path / "valid.txt"
    train_path.write_text("hello world\n" * 8, encoding="utf-8")
    valid_path.write_text("validation data\n" * 8, encoding="utf-8")
    return train_path, valid_path


@pytest.mark.integration
def test_train_lm_script_writes_run_artifacts(tmp_path, monkeypatch):
    train_lm_module = _load_script("train_lm.py")
    vocab_path, merge_path = _write_tokenizer_files(tmp_path)
    train_path, valid_path = _prepare_small_corpora(tmp_path)
    runs_root = tmp_path / "runs"
    config_path = _write_config(
        tmp_path,
        vocab_path=vocab_path,
        merge_path=merge_path,
        runs_root=runs_root,
        train_path=train_path,
        valid_path=valid_path,
    )
    monkeypatch.setattr(config_manager, "config_path", "config/default.yaml")

    exit_code = train_lm_module.main(["--config", str(config_path), "--run-name", "script-run"])

    run_dir = runs_root / "script-run"
    assert exit_code == 0
    assert (run_dir / "checkpoints" / "latest.pt").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "config.snapshot.yaml").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "tokenizer" / "vocab.txt").exists()
    assert (run_dir / "tokenizer" / "merges.txt").exists()

    with (run_dir / "summary.json").open("r", encoding="utf-8") as summary_file:
        summary = json.load(summary_file)

    assert summary["run_name"] == "script-run"
    assert summary["training"]["global_step"] == 2


@pytest.mark.integration
def test_evaluate_and_generate_scripts_use_saved_run_artifacts(tmp_path, monkeypatch):
    train_lm_module = _load_script("train_lm.py")
    evaluate_module = _load_script("evaluate_lm.py")
    generate_module = _load_script("generate_text.py")

    vocab_path, merge_path = _write_tokenizer_files(tmp_path)
    train_path, valid_path = _prepare_small_corpora(tmp_path)
    runs_root = tmp_path / "runs"
    config_path = _write_config(
        tmp_path,
        vocab_path=vocab_path,
        merge_path=merge_path,
        runs_root=runs_root,
        train_path=train_path,
        valid_path=valid_path,
    )
    monkeypatch.setattr(config_manager, "config_path", "config/default.yaml")

    train_lm_module.main(["--config", str(config_path), "--run-name", "script-run"])
    run_dir = runs_root / "script-run"

    evaluate_exit_code = evaluate_module.main(["--run-dir", str(run_dir)])
    generate_exit_code = generate_module.main(
        [
            "--run-dir",
            str(run_dir),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "256",
        ]
    )

    assert evaluate_exit_code == 0
    assert generate_exit_code == 0
    assert (run_dir / "eval" / "eval_owt.json").exists()
    assert (run_dir / "samples" / "sample_256.txt").exists()

    with (run_dir / "eval" / "eval_owt.json").open("r", encoding="utf-8") as eval_file:
        evaluation = json.load(eval_file)
    with (run_dir / "summary.json").open("r", encoding="utf-8") as summary_file:
        summary = json.load(summary_file)

    assert evaluation["dataset"] == str(valid_path)
    assert evaluation["loss"] >= 0.0
    assert evaluation["perplexity"] >= 1.0
    assert summary["generation"]["sample_256"]["generated_token_count"] == 256
    assert summary["evaluation"]["owt"]["dataset"] == str(valid_path)


@pytest.mark.integration
def test_run_pipeline_trains_evaluates_and_generates_from_scratch(tmp_path, monkeypatch):
    pipeline_module = _load_script("run_pipeline.py")
    train_path, valid_path = _prepare_small_corpora(tmp_path)
    runs_root = tmp_path / "runs"
    vocab_path = tmp_path / "artifacts" / "vocab.txt"
    merge_path = tmp_path / "artifacts" / "merges.txt"
    config_path = _write_config(
        tmp_path,
        vocab_path=vocab_path,
        merge_path=merge_path,
        runs_root=runs_root,
        train_path=train_path,
        valid_path=valid_path,
    )
    monkeypatch.setattr(config_manager, "config_path", "config/default.yaml")
    monkeypatch.chdir(Path(__file__).resolve().parents[2])

    exit_code = pipeline_module.main(
        [
            "--config",
            str(config_path),
            "--run-name",
            "pipeline-run",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "256",
        ]
    )

    run_dir = runs_root / "pipeline-run"
    assert exit_code == 0
    assert vocab_path.exists()
    assert merge_path.exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "eval" / "eval_owt.json").exists()
    assert (run_dir / "samples" / "sample_256.txt").exists()
