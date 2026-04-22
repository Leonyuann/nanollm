import math

import pytest
import torch

from config_manager import DataConfig
from model import DecoderLMConfig, DecoderOnlyTransformerLM
from tokenizer.tokenizer import tokenizer
from training import TrainingConfig, run_training


def _write_tokenizer_files(tmp_path):
    """Write a byte-level tokenizer artifact pair for integration tests.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        A ``(vocab_path, merge_path)`` tuple.
    """
    vocab_path = tmp_path / "vocab.txt"
    merge_path = tmp_path / "merges.txt"
    vocab_lines = [f"{idx}\t{idx:02x}" for idx in range(256)]
    vocab_path.write_text("\n".join(vocab_lines) + "\n", encoding="utf-8")
    merge_path.write_text("", encoding="utf-8")
    return vocab_path, merge_path


@pytest.mark.integration
def test_run_training_updates_model_parameters_and_reports_validation_loss(tmp_path):
    train_path = tmp_path / "train.txt"
    valid_path = tmp_path / "valid.txt"
    train_path.write_text("hello world\n" * 8, encoding="utf-8")
    valid_path.write_text("validation data\n" * 8, encoding="utf-8")

    vocab_path, merge_path = _write_tokenizer_files(tmp_path)
    text_tokenizer = tokenizer.from_files(str(vocab_path), str(merge_path), special_tokens=[])

    data_config = DataConfig(
        owt_train_path=str(train_path),
        owt_valid_path=str(valid_path),
        TinyStories_train_path=str(train_path),
        TinyStories_valid_path=str(valid_path),
    )
    model_config = DecoderLMConfig(
        vocab_size=256,
        max_seq_len=4,
        d_model=16,
        num_layers=2,
        num_heads=4,
        ffn_hidden_dim=32,
        norm_type="rmsnorm",
        ffn_type="swiglu",
        use_residual=True,
        dropout=0.0,
        tie_embeddings=False,
        rope_base=10000.0,
        bias=True,
    )
    training_config = TrainingConfig(
        dataset="tinystories",
        batch_size=2,
        max_steps=3,
        learning_rate=1e-3,
        warmup_steps=2,
        log_interval=1,
        eval_interval=1,
        eval_steps=1,
        device="cpu",
        seed=0,
    )

    model = DecoderOnlyTransformerLM(model_config)
    initial_embedding = model.token_embedding.weight.detach().clone()

    result = run_training(
        model=model,
        text_tokenizer=text_tokenizer,
        data_config=data_config,
        training_config=training_config,
    )

    assert result.global_step == 3
    assert math.isfinite(result.train_loss)
    assert result.validation_loss is not None
    assert math.isfinite(result.validation_loss)
    assert not torch.allclose(model.token_embedding.weight.detach(), initial_embedding)
