from itertools import islice
from pathlib import Path

import argparse
import numpy as np
from tqdm import tqdm

from config_manager import load_BPEConfig, load_DataConfig
from tokenizer.tokenizer import tokenizer

DTYPE = np.dtype(np.uint16)
CHUNK_SIZE = 1_000_000

def parse_args() ->argparse.Namespace:
    bpeconfig = load_BPEConfig()
    dataconfig = load_DataConfig()

    parser =argparse.ArgumentParser(description="Tokenize target text into target training data.")

    # Tokenizer configure
    parser.add_argument("--vocab_filepath", type=str, default=bpeconfig.vocab_path)
    parser.add_argument("--merge_filepath", type=str, default=bpeconfig.merge_path)
    parser.add_argument("--special_tokens", type=list[str], default=bpeconfig.special_tokens)

    # Input and output data configure
    parser.add_argument("--input_filepath", type=str, default=dataconfig.TinyStories_train_path)
    parser.add_argument("--output_filepath", type=str, default=dataconfig.training_data_path)

    return parser.parse_args()


def tokenize_file(input_path: str, output_path: str, tokeni: tokenizer) -> int:
    input_path, output_path = Path(input_path), Path(output_path)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if max(tokeni.vocab, default=0) > np.iinfo(DTYPE).max:
        raise ValueError(f"Token IDs exceed {DTYPE}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_count = 0

    def lines(input_file, progress):
        for line in input_file:
            progress.update(len(line))
            yield line.decode("utf-8")

    try:
        with (
            input_path.open("rb") as input_file,
            temp_path.open("wb") as output_file,
            tqdm(
                total=input_path.stat().st_size,
                desc="Tokenizing",
                unit="B",
                unit_scale=True,
            ) as progress,
        ):
            tokens = tokeni.encode_iterable(lines(input_file, progress))
            while True:
                chunk = np.fromiter(islice(tokens, CHUNK_SIZE), dtype=DTYPE)
                if not chunk.size:
                    break
                chunk.tofile(output_file)
                token_count += chunk.size
                progress.set_postfix(tokens=f"{token_count:,}")

        temp_path.replace(output_path)
        return token_count
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def main(args) -> None:
    tokeni = tokenizer.from_files(
        vocab_filepath=args.vocab_filepath,
        merge_filepath=args.merge_filepath,
        special_tokens=args.special_tokens,
    )

    token_count = tokenize_file(
        input_path=args.input_filepath,
        output_path=args.output_filepath,
        tokeni=tokeni,
    )
    print(f"Wrote {token_count:,} tokens to {args.output_filepath}")


if __name__ == "__main__":

    args = parse_args()
    main(args)
