from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import config_manager
from config_manager import load_BPEConfig
from tokenizer import train_bpe
from tokenizer.types import Merges, Vocabulary


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for tokenizer training.

    Args:
        None.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Train the byte-level BPE tokenizer and write machine-readable artifacts.",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the tokenizer training corpus.",
    )
    return parser


def write_vocab_file(
    output_path: Path,
    vocab: Vocabulary,
) -> None:
    """Write vocabulary artifacts in the format expected by tokenizer.from_files.

    Args:
        output_path: Destination path for the serialized vocabulary file.
        vocab: Vocabulary mapping token ids to byte tokens.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as vocab_file:
        for token_id, byte_token in sorted(vocab.items()):
            vocab_file.write(f"{token_id}\t{byte_token.hex()}\n")


def write_merge_file(
    output_path: Path,
    merges: Merges,
) -> None:
    """Write merge artifacts in the format expected by tokenizer.from_files.

    Args:
        output_path: Destination path for the serialized merge file.
        merges: Learned merge pairs in training order.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as merge_file:
        for left, right in merges:
            merge_file.write(f"{left.hex()}\t{right.hex()}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Train the tokenizer from a corpus and save its artifacts.

    Args:
        argv: Optional CLI-style argument sequence. When omitted, argparse reads
            from the process command line.

    Returns:
        Process exit code.
    """
    args = build_argument_parser().parse_args(argv)
    config_manager.config_path = str(Path(args.config))

    bpe_config = load_BPEConfig()
    vocab, merges = train_bpe.train_bpe(
        input_path=args.input,
        vocab_size=bpe_config.vocab_size,
        special_tokens=bpe_config.special_tokens,
    )

    vocab_path = Path(bpe_config.vocab_path)
    merge_path = Path(bpe_config.merge_path)
    write_vocab_file(vocab_path, vocab)
    write_merge_file(merge_path, merges)

    print(
        "Tokenizer artifacts written "
        f"(vocab_size={len(vocab)}, merges={len(merges)}) "
        f"to {vocab_path} and {merge_path}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
