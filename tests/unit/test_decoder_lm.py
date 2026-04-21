from dataclasses import asdict

import pytest
import torch

import config_manager
from model import DecoderLMConfig, DecoderOnlyTransformerLM


def _model_config(**overrides) -> DecoderLMConfig:
    defaults = {
        "vocab_size": 32,
        "max_seq_len": 16,
        "d_model": 16,
        "num_layers": 2,
        "num_heads": 4,
        "ffn_hidden_dim": 64,
        "norm_type": "rmsnorm",
        "ffn_type": "swiglu",
        "use_residual": True,
        "dropout": 0.0,
        "tie_embeddings": True,
        "rope_base": 10000.0,
        "bias": True,
    }
    defaults.update(overrides)
    return DecoderLMConfig(**defaults)


@pytest.mark.unit
def test_forward_returns_logits_with_expected_shape_and_dtype():
    torch.manual_seed(0)
    model = DecoderOnlyTransformerLM(_model_config(vocab_size=41, max_seq_len=8))

    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    logits = model(input_ids)

    assert logits.shape == (2, 4, 41)
    assert logits.dtype == torch.float32


@pytest.mark.unit
def test_forward_is_strictly_causal_for_prefix_positions():
    torch.manual_seed(0)
    model = DecoderOnlyTransformerLM(_model_config(vocab_size=23, max_seq_len=8))
    model.eval()

    prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)
    extended = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    prefix_logits = model(prefix)
    extended_logits = model(extended)

    assert torch.allclose(
        prefix_logits,
        extended_logits[:, : prefix.size(1), :],
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.unit
@pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm", "none"])
def test_forward_supports_all_norm_types(norm_type):
    model = DecoderOnlyTransformerLM(_model_config(norm_type=norm_type))

    logits = model(torch.tensor([[0, 1, 2, 3]], dtype=torch.long))

    assert logits.shape == (1, 4, 32)


@pytest.mark.unit
@pytest.mark.parametrize("ffn_type", ["swiglu", "gelu", "silu"])
def test_forward_supports_all_ffn_types(ffn_type):
    model = DecoderOnlyTransformerLM(_model_config(ffn_type=ffn_type))

    logits = model(torch.tensor([[0, 1, 2, 3]], dtype=torch.long))

    assert logits.shape == (1, 4, 32)


@pytest.mark.unit
def test_forward_supports_disabling_residual_connections():
    model = DecoderOnlyTransformerLM(_model_config(use_residual=False))

    logits = model(torch.tensor([[0, 1, 2, 3]], dtype=torch.long))

    assert logits.shape == (1, 4, 32)


@pytest.mark.unit
def test_tied_embeddings_share_the_same_weight_parameter():
    model = DecoderOnlyTransformerLM(_model_config(tie_embeddings=True))

    assert model.token_embedding.weight is model.lm_head.weight


@pytest.mark.unit
def test_untied_embeddings_use_distinct_weight_parameters():
    model = DecoderOnlyTransformerLM(_model_config(tie_embeddings=False))

    assert model.token_embedding.weight is not model.lm_head.weight


@pytest.mark.unit
def test_forward_rejects_sequences_longer_than_max_seq_len():
    model = DecoderOnlyTransformerLM(_model_config(max_seq_len=4))

    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long))


@pytest.mark.unit
def test_load_decoder_lm_config_reads_default_yaml_model_section():
    model_config = config_manager.load_DecoderLMConfig()

    assert isinstance(model_config, DecoderLMConfig)
    assert asdict(model_config) == {
        "vocab_size": 10000,
        "max_seq_len": 256,
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 4,
        "ffn_hidden_dim": 31375,
        "norm_type": "rmsnorm",
        "ffn_type": "swiglu",
        "use_residual": True,
        "dropout": 0.0,
        "tie_embeddings": True,
        "rope_base": 10000.0,
        "bias": True,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"d_model": 18, "num_heads": 4}, "d_model must be divisible by num_heads"),
        ({"d_model": 15, "num_heads": 3}, "RoPE requires an even head dimension"),
        ({"ffn_hidden_dim": 0}, "ffn_hidden_dim must be positive"),
        ({"dropout": -0.1}, "dropout must be between 0.0 and 1.0 inclusive"),
        ({"norm_type": "batchnorm"}, "norm_type must be one of"),
        ({"ffn_type": "relu"}, "ffn_type must be one of"),
    ],
)
def test_decoder_lm_config_rejects_invalid_values(overrides, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        _model_config(**overrides)


@pytest.mark.unit
def test_default_model_config_stays_within_50m_parameter_budget():
    model = DecoderOnlyTransformerLM(config_manager.load_DecoderLMConfig())

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    assert parameter_count == 49_998_856
    assert parameter_count <= 50_000_000
