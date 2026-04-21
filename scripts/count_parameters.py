from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from model import DecoderLMConfig, DecoderOnlyTransformerLM


def load_model_config(config_path: Path) -> DecoderLMConfig:
    with config_path.open("r", encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)
    return DecoderLMConfig(**config_data["model"])


def count_trainable_parameters(model: DecoderOnlyTransformerLM) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count trainable parameters for the decoder-only Transformer LM.",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to a YAML config file containing a model section.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    config_path = Path(args.config)
    model_config = load_model_config(config_path)
    model = DecoderOnlyTransformerLM(model_config)
    total_parameters = count_trainable_parameters(model)

    print(f"config={config_path}")
    print(f"trainable_parameters={total_parameters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
