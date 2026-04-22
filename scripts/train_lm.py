from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

import config_manager
from config_manager import load_ArtifactsConfig
from model import DecoderOnlyTransformerLM
from run_artifacts import (
    build_run_artifacts,
    copy_tokenizer_artifacts,
    update_summary,
    write_config_snapshot,
)
from tokenizer.tokenizer import tokenizer
from training import run_training


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for language-model training.

    Args:
        None.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Train the decoder-only Transformer LM with AdamW and warmup.",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. Defaults to a timestamp-based name.",
    )
    return parser


def build_default_run_name() -> str:
    """Create a timestamp-based run name.

    Args:
        None.

    Returns:
        A timestamp-based run name string.
    """
    return datetime.now().strftime("run-%Y%m%d-%H%M%S")


def main(argv: Sequence[str] | None = None) -> int:
    """Load config, build training objects, run training, and save artifacts.

    Args:
        argv: Optional CLI-style argument sequence.

    Returns:
        Process exit code.
    """
    args = build_argument_parser().parse_args(argv)
    config_path = Path(args.config)
    config_manager.config_path = str(config_path)

    artifacts_config = load_ArtifactsConfig()
    bpe_config = config_manager.load_BPEConfig()
    data_config = config_manager.load_DataConfig()
    model_config = config_manager.load_DecoderLMConfig()
    training_config = config_manager.load_TrainingConfig()
    run_name = args.run_name or build_default_run_name()
    run_artifacts = build_run_artifacts(artifacts_config.runs_root, run_name)

    write_config_snapshot(config_path, run_artifacts.config_snapshot_path)
    copied_vocab_path, copied_merge_path = copy_tokenizer_artifacts(
        bpe_config.vocab_path,
        bpe_config.merge_path,
        run_artifacts.tokenizer_dir,
    )

    text_tokenizer = tokenizer.from_files(
        vocab_filepath=bpe_config.vocab_path,
        merge_filepath=bpe_config.merge_path,
        special_tokens=bpe_config.special_tokens,
    )
    model = DecoderOnlyTransformerLM(model_config)

    training_result = run_training(
        model=model,
        text_tokenizer=text_tokenizer,
        data_config=data_config,
        training_config=training_config,
        artifact_dir=run_artifacts.run_dir,
    )
    update_summary(
        run_artifacts.summary_path,
        {
            "config_snapshot_path": str(run_artifacts.config_snapshot_path),
            "run_dir": str(run_artifacts.run_dir),
            "run_name": run_name,
            "tokenizer": {
                "merge_path": str(copied_merge_path),
                "vocab_path": str(copied_vocab_path),
            },
            "training": {
                "best_checkpoint_path": str(
                    run_artifacts.checkpoints_dir / "best.pt"
                ),
                "best_validation_loss": training_result.best_validation_loss,
                "global_step": training_result.global_step,
                "latest_checkpoint_path": str(
                    run_artifacts.checkpoints_dir / "latest.pt"
                ),
                "metrics_path": str(run_artifacts.metrics_path),
                "train_loss": training_result.train_loss,
                "validation_loss": training_result.validation_loss,
            },
        },
    )
    print(f"run_dir={run_artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
