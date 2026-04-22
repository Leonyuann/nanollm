from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import config_manager
from config_manager import DataConfig, DecoderLMConfig
from model import DecoderOnlyTransformerLM
from run_artifacts import resolve_checkpoint_path, resolve_run_dir, update_summary
from tokenizer.tokenizer import tokenizer
from training.data import TokenSequenceIterableDataset
from training.train import resolve_device


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for standalone evaluation.

    Args:
        None.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a saved LM checkpoint on a validation corpus.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory containing checkpoints, config snapshot, and tokenizer.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint selector: best, latest, or an explicit checkpoint path.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional validation corpus path. Defaults to the configured OWT validation path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional evaluation output path. Defaults to <run-dir>/eval/eval_owt.json.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device selection mode: auto, cpu, or cuda.",
    )
    return parser


def load_run_tokenizer(run_dir: Path) -> tokenizer:
    """Load the copied tokenizer artifacts for a run.

    Args:
        run_dir: Run directory containing the tokenizer subdirectory.

    Returns:
        The runtime tokenizer for the run.
    """
    bpe_config = config_manager.load_BPEConfig()
    return tokenizer.from_files(
        vocab_filepath=str(run_dir / "tokenizer" / "vocab.txt"),
        merge_filepath=str(run_dir / "tokenizer" / "merges.txt"),
        special_tokens=bpe_config.special_tokens,
    )


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> DecoderOnlyTransformerLM:
    """Restore a language model from a saved checkpoint.

    Args:
        checkpoint_path: Saved checkpoint path.
        device: Device on which the model should be loaded.

    Returns:
        The restored language model in evaluation mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = DecoderLMConfig(**checkpoint["model_config"])
    model = DecoderOnlyTransformerLM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def evaluate_corpus(
    model: DecoderOnlyTransformerLM,
    text_tokenizer: tokenizer,
    dataset_path: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate a model on an entire text corpus.

    Args:
        model: Language model under evaluation.
        text_tokenizer: Tokenizer used for encoding the corpus.
        dataset_path: Validation corpus path.
        batch_size: Evaluation batch size.
        device: Device used for evaluation.

    Returns:
        A dictionary containing mean loss, perplexity, batch count, and token count.

    Raises:
        ValueError: If the corpus yields no evaluation batches.
    """
    dataset = TokenSequenceIterableDataset(
        file_path=dataset_path,
        text_tokenizer=text_tokenizer,
        seq_len=model.config.max_seq_len,
        repeat=False,
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    total_loss = 0.0
    total_tokens = 0
    batch_count = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(input_ids)
            batch_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += batch_loss.item()
            total_tokens += targets.numel()
            batch_count += 1

    if batch_count == 0:
        raise ValueError("evaluation dataset did not yield any batches")

    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "num_batches": batch_count,
        "num_tokens": total_tokens,
        "perplexity": math.exp(mean_loss),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run standalone checkpoint evaluation and save its results.

    Args:
        argv: Optional CLI-style argument sequence.

    Returns:
        Process exit code.
    """
    args = build_argument_parser().parse_args(argv)
    run_dir = resolve_run_dir(args.run_dir, args.checkpoint)
    checkpoint_path = resolve_checkpoint_path(run_dir, args.checkpoint)
    config_snapshot_path = run_dir / "config.snapshot.yaml"
    config_manager.config_path = str(config_snapshot_path)

    data_config = config_manager.load_DataConfig()
    training_config = config_manager.load_TrainingConfig()
    dataset_path = args.dataset_path or data_config.owt_valid_path
    output_path = Path(args.output) if args.output is not None else run_dir / "eval" / "eval_owt.json"
    device = resolve_device(args.device)

    text_tokenizer = load_run_tokenizer(run_dir)
    model = load_model_from_checkpoint(checkpoint_path, device)
    metrics = evaluate_corpus(
        model=model,
        text_tokenizer=text_tokenizer,
        dataset_path=dataset_path,
        batch_size=training_config.batch_size,
        device=device,
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        **metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=True, sort_keys=True)
        output_file.write("\n")

    update_summary(
        run_dir / "summary.json",
        {
            "evaluation": {
                "owt": {
                    "checkpoint": str(checkpoint_path),
                    "dataset": str(dataset_path),
                    "loss": result["loss"],
                    "output_path": str(output_path),
                    "perplexity": result["perplexity"],
                }
            }
        },
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
