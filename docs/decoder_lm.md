# Decoder-Only Transformer LM

## Overview

This project now includes a compact decoder-only Transformer language model in `src/model/`.

The implementation is intentionally small and explicit:

- token embeddings only, with no separate learned positional embedding
- stacked decoder blocks with causal self-attention
- RoPE applied to attention queries and keys
- configurable normalization and FFN variants for ablation-style experiments
- final projection to next-token logits

The default configuration is:

- `RoPE`
- `RMSNorm`
- `SwiGLU`
- residual connections enabled
- `dropout = 0.0`
- tied token embedding and LM head weights

## Module Map And Data Flow

### `src/model/config.py`

Defines `DecoderLMConfig`, the validated configuration object for model construction.

### `src/model/decoder_lm.py`

Contains the full model stack:

- `RMSNorm`
- `RotaryEmbedding`
- `FeedForward`
- `CausalSelfAttention`
- `TransformerBlock`
- `DecoderOnlyTransformerLM`

### Data flow

The model executes the following pipeline:

1. `input_ids` with shape `[batch, seq_len]`
2. token embedding lookup to `[batch, seq_len, d_model]`
3. embedding dropout
4. repeated decoder blocks
5. final normalization
6. LM head projection to `[batch, seq_len, vocab_size]`

Each decoder block applies:

1. pre-norm
2. causal self-attention
3. residual merge or direct replacement
4. pre-norm
5. FFN
6. residual merge or direct replacement

## Configuration Fields

Model configuration is loaded from `config/default.yaml` through `config_manager.load_DecoderLMConfig()`.

```yaml
model:
  vocab_size: 10000
  max_seq_len: 256
  d_model: 512
  num_layers: 10
  num_heads: 8
  ffn_hidden_dim: 2048
  norm_type: "rmsnorm"
  ffn_type: "swiglu"
  use_residual: true
  dropout: 0.0
  tie_embeddings: true
  rope_base: 10000.0
  bias: true
```

Field meanings:

- `vocab_size`: output vocabulary size and embedding table size
- `max_seq_len`: maximum sequence length accepted by `forward(...)`
- `d_model`: hidden width of the model
- `num_layers`: number of decoder blocks
- `num_heads`: number of attention heads
- `ffn_hidden_dim`: intermediate width used by the FFN
- `norm_type`: one of `rmsnorm`, `layernorm`, or `none`
- `ffn_type`: one of `swiglu`, `gelu`, or `silu`
- `use_residual`: toggles residual additions around attention and FFN
- `dropout`: dropout probability used in embeddings, attention, and FFN
- `tie_embeddings`: shares token embedding and LM head weights when enabled
- `rope_base`: RoPE frequency base
- `bias`: toggles bias terms in linear layers

The config is validated eagerly. Invalid enum values, illegal dropout values, non-divisible head dimensions, and odd RoPE head dimensions raise `ValueError` during construction.

The repository also includes a small helper script for counting trainable parameters:

```bash
uv run python scripts/count_parameters.py
```

With the current default config, the model has `49,998,856` trainable parameters, which stays within the `50,000,000` cap.

## Tensor Shape Conventions

The public forward API is:

```python
logits = model(input_ids)
```

Shapes:

- input: `input_ids` is `[batch, seq_len]`
- embeddings: `[batch, seq_len, d_model]`
- q, k, v inside attention: `[batch, num_heads, seq_len, head_dim]`
- output logits: `[batch, seq_len, vocab_size]`

The model returns logits for every position. The caller is responsible for shifting targets for next-token training.

## RoPE And Causal Attention

The model does not use a separate positional embedding table. Position information is injected inside attention with rotary position embedding:

- queries and keys are split into heads
- RoPE phases are generated from token positions and `rope_base`
- the rotation is applied to the query and key head dimensions

Causality is enforced with:

```python
torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)
```

This guarantees that token `t` can only attend to positions `<= t`.

## Switchable Ablation Knobs

### Normalization

- `rmsnorm`: custom RMSNorm implementation
- `layernorm`: `torch.nn.LayerNorm`
- `none`: `torch.nn.Identity`

### FFN

- `swiglu`: gated FFN using `SiLU(gate) * value`
- `gelu`: standard 2-layer GELU MLP
- `silu`: standard 2-layer SiLU MLP

### Residuals

When `use_residual=true`, each sublayer update is:

```python
x = x + sublayer(norm(x))
```

When `use_residual=false`, each sublayer update is:

```python
x = sublayer(norm(x))
```

## Minimal Usage Example

```python
import torch

from config_manager import load_DecoderLMConfig
from model import DecoderOnlyTransformerLM

config = load_DecoderLMConfig()
model = DecoderOnlyTransformerLM(config)

input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
logits = model(input_ids)
print(logits.shape)  # [1, 4, vocab_size]
```

## Unsupported In V1

The current implementation intentionally does not include:

- training loop or optimizer setup
- loss computation helper
- text generation or sampling utilities
- KV cache for incremental decoding
- returning hidden states or attention maps
- flash-attention-specific tuning beyond PyTorch SDPA defaults
- custom parameter initialization schemes
