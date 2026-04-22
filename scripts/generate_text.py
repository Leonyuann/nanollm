from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

import config_manager
from config_manager import DecoderLMConfig
from model import DecoderOnlyTransformerLM
from run_artifacts import resolve_checkpoint_path, resolve_run_dir, update_summary
from tokenizer.tokenizer import tokenizer
from training.train import resolve_device


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for greedy text generation.

    Args:
        None.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate text from a saved LM checkpoint with greedy decoding.",
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
        "--prompt",
        default="Once upon a time",
        help="Prompt text used to seed generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional sample output path. Defaults to <run-dir>/samples/sample_256.txt.",
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


def generate_ids(
    model: DecoderOnlyTransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    """Generate new token ids with greedy decoding.

    Args:
        model: Language model used for generation.
        prompt_ids: Prompt token ids.
        max_new_tokens: Number of new tokens to generate.
        device: Device used for inference.

    Returns:
        The complete prompt-plus-generated token sequence.

    Raises:
        ValueError: If ``prompt_ids`` is empty or ``max_new_tokens`` is not positive.
    """
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    generated_ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_ids = generated_ids[-model.config.max_seq_len :]
            input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
            logits = model(input_ids)
            next_token_id = int(torch.argmax(logits[0, -1]).item())
            generated_ids.append(next_token_id)
    return generated_ids


def main(argv: Sequence[str] | None = None) -> int:
    """Run greedy generation from a saved checkpoint and save the sample.

    Args:
        argv: Optional CLI-style argument sequence.

    Returns:
        Process exit code.
    """
    args = build_argument_parser().parse_args(argv)
    run_dir = resolve_run_dir(args.run_dir, args.checkpoint)
    checkpoint_path = resolve_checkpoint_path(run_dir, args.checkpoint)
    config_manager.config_path = str(run_dir / "config.snapshot.yaml")

    device = resolve_device(args.device)
    text_tokenizer = load_run_tokenizer(run_dir)
    model = load_model_from_checkpoint(checkpoint_path, device)
    prompt_ids = text_tokenizer.encode(args.prompt)
    generated_ids = generate_ids(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    completion_ids = generated_ids[len(prompt_ids) :]
    completion_text = text_tokenizer.decode(completion_ids)
    output_path = Path(args.output) if args.output is not None else run_dir / "samples" / "sample_256.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write("Prompt:\n")
        output_file.write(args.prompt)
        output_file.write("\n\nCompletion:\n")
        output_file.write(completion_text)
        output_file.write("\n")

    update_summary(
        run_dir / "summary.json",
        {
            "generation": {
                "sample_256": {
                    "checkpoint": str(checkpoint_path),
                    "generated_token_count": len(completion_ids),
                    "output_path": str(output_path),
                    "prompt": args.prompt,
                    "prompt_token_count": len(prompt_ids),
                }
            }
        },
    )
    print(f"sample_path={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
