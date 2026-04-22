from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import config_manager


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the end-to-end project pipeline.

    Args:
        None.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run tokenizer training, LM training, evaluation, and generation.",
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
    parser.add_argument(
        "--prompt",
        default="Once upon a time",
        help="Prompt text used for the final generation sample.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Number of new tokens to generate in the final sample.",
    )
    return parser


def build_default_run_name() -> str:
    """Create a timestamp-based run name for pipeline execution.

    Args:
        None.

    Returns:
        A timestamp-based run name string.
    """
    return datetime.now().strftime("run-%Y%m%d-%H%M%S")


def run_script(arguments: list[str]) -> None:
    """Execute another repository script with the current Python interpreter.

    Args:
        arguments: Script path followed by its CLI arguments.

    Returns:
        None.

    Raises:
        subprocess.CalledProcessError: If the child script fails.
    """
    subprocess.run([sys.executable, *arguments], check=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the minimal end-to-end project pipeline.

    Args:
        argv: Optional CLI-style argument sequence.

    Returns:
        Process exit code.
    """
    args = build_argument_parser().parse_args(argv)
    config_manager.config_path = str(Path(args.config))

    bpe_config = config_manager.load_BPEConfig()
    data_config = config_manager.load_DataConfig()
    run_name = args.run_name or build_default_run_name()

    if not Path(bpe_config.vocab_path).exists() or not Path(bpe_config.merge_path).exists():
        run_script(
            [
                "scripts/train_tokenizer.py",
                "--config",
                args.config,
                "--input",
                data_config.owt_train_path,
            ]
        )

    run_script(
        [
            "scripts/train_lm.py",
            "--config",
            args.config,
            "--run-name",
            run_name,
        ]
    )
    run_dir = config_manager.load_ArtifactsConfig().runs_root
    run_path = Path(run_dir) / run_name

    run_script(
        [
            "scripts/evaluate_lm.py",
            "--run-dir",
            str(run_path),
        ]
    )
    run_script(
        [
            "scripts/generate_text.py",
            "--run-dir",
            str(run_path),
            "--prompt",
            args.prompt,
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
    )
    print(f"run_dir={run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
