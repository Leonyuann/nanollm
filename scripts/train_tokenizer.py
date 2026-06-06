import argparse
from pathlib import Path

from tokenizer import train_bpe
from config_manager import load_BPEConfig, load_DataConfig


def parse_args() -> argparse.Namespace:
    """Parse tokenizer training command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer. Uses OWT training data by default."
    )

    bpeconfig = load_BPEConfig()
    dataconfig = load_DataConfig()

    parser.add_argument("--input_path", type=str, default=dataconfig.owt_train_path)
    parser.add_argument("--vocab_size", type=int, default=bpeconfig.vocab_size)
    parser.add_argument("--special_tokens", nargs="+", default=bpeconfig.special_tokens)

    parser.add_argument("--target_merge_path", type=str, default=bpeconfig.merge_path)
    parser.add_argument("--target_vocab_path", type=str, default=bpeconfig.vocab_path)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Train and write tokenizer artifacts through temporary files.

    Args:
        args: Parsed tokenizer training arguments.
    """
    vocab, merges = train_bpe.train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        show_progress=True,
    )

    merge_path = Path(args.target_merge_path)
    temp_merge_path = merge_path.with_suffix(merge_path.suffix + ".tmp")

    vocab_path = Path(args.target_vocab_path)
    temp_vocab_path = vocab_path.with_suffix(vocab_path.suffix + ".tmp")

    merge_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(temp_vocab_path, "w", encoding="utf-8") as f:
            for key in vocab:
                f.write(f"{key}\t{vocab[key].hex()}\n")

        with open(temp_merge_path, "w", encoding="utf-8") as f:
            for bytes1, bytes2 in merges:
                f.write(f"{bytes1.hex()}\t{bytes2.hex()}\n")

        temp_merge_path.replace(merge_path)
        temp_vocab_path.replace(vocab_path)

    except BaseException:
        temp_merge_path.unlink(missing_ok=True)
        temp_vocab_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main(parse_args())
