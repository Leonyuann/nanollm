from __future__ import annotations

import argparse
from pathlib import Path

import config_manager
from model import DecoderOnlyTransformerLM
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
    return parser


def main() -> int:
    """Load config, build training objects, and run LM training.

    Args:
        None.

    Returns:
        Process exit code.
    """
    args = build_argument_parser().parse_args()
    config_manager.config_path = str(Path(args.config))

    bpe_config = config_manager.load_BPEConfig()
    data_config = config_manager.load_DataConfig()
    model_config = config_manager.load_DecoderLMConfig()
    training_config = config_manager.load_TrainingConfig()

    text_tokenizer = tokenizer.from_files(
        vocab_filepath=bpe_config.vocab_path,
        merge_filepath=bpe_config.merge_path,
        special_tokens=bpe_config.special_tokens,
    )
    model = DecoderOnlyTransformerLM(model_config)

    run_training(
        model=model,
        text_tokenizer=text_tokenizer,
        data_config=data_config,
        training_config=training_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
