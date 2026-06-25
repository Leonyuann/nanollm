import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for JSONL sampling.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Sample a fixed number of records from a JSONL file."
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_num", type=int, default=20)
    parser.add_argument("--input_file", type=str, default="data/cc_sample.jsonl")
    parser.add_argument("--output_file", type=str, default="data/cc_sample_sampled.jsonl")

    return parser.parse_args()


def sample_jsonl(
    input_file: Path,
    output_file: Path,
    sample_num: int,
    seed: int,
) -> int:
    """Sample records from a JSONL file without loading it fully into memory.

    Uses reservoir sampling so every valid JSONL record has equal probability
    of being selected. Blank lines are ignored, and sampled records are written
    in their original input order.

    Args:
        input_file: Source JSONL file.
        output_file: Destination JSONL file.
        sample_num: Number of records to sample.
        seed: Random seed for reproducible sampling.

    Returns:
        Number of sampled records written.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If sample_num is not positive, the input file has too few
            records, or a line is not valid JSON.
    """
    if sample_num <= 0:
        raise ValueError("--sample_num must be a positive integer.")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")

    rng = random.Random(seed)
    reservoir: list[tuple[int, object]] = []
    valid_record_count = 0

    with (
        input_file.open("rb") as source,
        tqdm(
            total=input_file.stat().st_size,
            desc="Sampling",
            unit="B",
            unit_scale=True,
        ) as progress,
    ):
        for line_number, raw_line in enumerate(source, start=1):
            progress.update(len(raw_line))
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {input_file}:{line_number}: {exc.msg}"
                ) from exc

            valid_record_count += 1
            item = (line_number, record)

            if len(reservoir) < sample_num:
                reservoir.append(item)
                continue

            replacement_index = rng.randrange(valid_record_count)
            if replacement_index < sample_num:
                reservoir[replacement_index] = item

    if valid_record_count < sample_num:
        raise ValueError(
            f"Requested {sample_num} samples, but only found "
            f"{valid_record_count} valid records in {input_file}."
        )

    try:
        with temp_file.open("w", encoding="utf-8") as target:
            for _, record in sorted(reservoir, key=lambda item: item[0]):
                json.dump(record, target, ensure_ascii=False)
                target.write("\n")
        temp_file.replace(output_file)
    except BaseException:
        temp_file.unlink(missing_ok=True)
        raise

    return len(reservoir)


def main(args: argparse.Namespace) -> None:
    """Run JSONL sampling from parsed command-line arguments.

    Args:
        args: Parsed command-line arguments.
    """
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    sampled_count = sample_jsonl(
        input_file=input_file,
        output_file=output_file,
        sample_num=args.sample_num,
        seed=args.seed,
    )
    print(f"Wrote {sampled_count:,} sampled records to {output_file}")


if __name__ == "__main__":
    main(parse_args())
