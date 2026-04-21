"""Decoder-only Transformer language model components.

This module implements a compact causal language model stack with:

- token embeddings
- rotary position embedding (RoPE) inside attention
- configurable normalization and FFN variants
- stacked decoder blocks
- a final projection to vocabulary logits
"""

import torch
from torch import nn
from torch.nn import functional as F

from .config import DecoderLMConfig


class RMSNorm(nn.Module):
    """Root mean square normalization without mean subtraction.

    Attributes:
        eps: Small constant added to the denominator for numerical stability.
        weight: Learnable per-channel scaling parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension by its root mean square.

        Args:
            x: Input tensor whose last dimension is the hidden dimension.

        Returns:
            The normalized tensor with the same shape as ``x``.
        """
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(mean_square + self.eps)
        return normalized * self.weight


def _build_norm(dim: int, norm_type: str) -> nn.Module:
    """Construct the normalization layer requested by the config.

    Args:
        dim: Hidden dimension to normalize.
        norm_type: Normalization implementation name.

    Returns:
        The instantiated normalization module.

    Raises:
        ValueError: If ``norm_type`` is not supported.
    """
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {norm_type}")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate even and odd features into RoPE's paired representation.

    Args:
        x: Tensor whose last dimension is organized as even/odd feature pairs.

    Returns:
        Tensor with each feature pair rotated by 90 degrees.
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embedding to a head-projected tensor.

    Args:
        x: Query or key tensor with shape ``[batch, heads, seq_len, head_dim]``.
        cos: Cosine phases broadcastable to ``x``.
        sin: Sine phases broadcastable to ``x``.

    Returns:
        Tensor with RoPE applied along the head dimension.
    """
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """Generate RoPE cosine and sine tables for a sequence length.

    Attributes:
        inv_freq: Inverse frequencies used to build rotary phases.
    """

    def __init__(self, head_dim: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create cosine and sine tables for the requested sequence length.

        Args:
            seq_len: Number of positions that need rotary phases.
            device: Target device for the returned tensors.
            dtype: Target dtype for the returned tensors.

        Returns:
            A tuple ``(cos, sin)`` broadcastable to attention queries and keys.
        """
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)
        cos = cos.unsqueeze(0).unsqueeze(0).to(dtype)
        sin = sin.unsqueeze(0).unsqueeze(0).to(dtype)
        return cos, sin


class FeedForward(nn.Module):
    """Position-wise feed-forward network used inside each decoder block.

    The hidden projection can implement a standard MLP or a gated SwiGLU
    variant depending on ``config.ffn_type``.

    Attributes:
        ffn_type: Feed-forward activation variant selected by configuration.
        hidden_dropout: Dropout applied after the nonlinearity or gating step.
        output_dropout: Dropout applied after the output projection.
    """

    def __init__(self, config: DecoderLMConfig):
        super().__init__()
        self.ffn_type = config.ffn_type
        self.hidden_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

        if self.ffn_type == "swiglu":
            self.up_proj = nn.Linear(
                config.d_model,
                2 * config.ffn_hidden_dim,
                bias=config.bias,
            )
        else:
            self.up_proj = nn.Linear(
                config.d_model,
                config.ffn_hidden_dim,
                bias=config.bias,
            )

        self.down_proj = nn.Linear(
            config.ffn_hidden_dim,
            config.d_model,
            bias=config.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform hidden states independently at each sequence position.

        Args:
            x: Input tensor with shape ``[batch, seq_len, d_model]``.

        Returns:
            Tensor with the same shape as ``x`` after FFN processing.
        """
        hidden = self.up_proj(x)

        if self.ffn_type == "swiglu":
            gate, value = hidden.chunk(2, dim=-1)
            hidden = F.silu(gate) * value
        elif self.ffn_type == "gelu":
            hidden = F.gelu(hidden)
        elif self.ffn_type == "silu":
            hidden = F.silu(hidden)
        else:
            raise ValueError(f"Unsupported FFN type: {self.ffn_type}")

        hidden = self.hidden_dropout(hidden)
        hidden = self.down_proj(hidden)
        return self.output_dropout(hidden)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and causal masking.

    Attributes:
        d_model: Model hidden width.
        num_heads: Number of attention heads.
        head_dim: Per-head hidden width.
        attn_dropout: Dropout probability applied inside attention.
        qkv_proj: Shared projection producing queries, keys, and values.
        out_proj: Output projection after concatenating attention heads.
        resid_dropout: Dropout applied to the projected attention output.
        rotary: RoPE helper for query and key position encoding.
    """

    def __init__(self, config: DecoderLMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.attn_dropout = config.dropout

        self.qkv_proj = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=config.bias,
        )
        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.bias,
        )
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run causal self-attention over a batch of hidden states.

        Args:
            x: Input tensor with shape ``[batch, seq_len, d_model]``.

        Returns:
            Tensor with shape ``[batch, seq_len, d_model]``.
        """
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split the projected states into attention heads.
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Inject position information directly into queries and keys.
        cos, sin = self.rotary(seq_len, device=q.device, dtype=q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        attn_output = self.out_proj(attn_output)
        return self.resid_dropout(attn_output)


class TransformerBlock(nn.Module):
    """Pre-norm decoder block with attention and feed-forward sublayers.

    Attributes:
        use_residual: Whether sublayer outputs are added back to their inputs.
        attn_norm: Normalization applied before self-attention.
        ffn_norm: Normalization applied before the feed-forward network.
        attn: Causal self-attention sublayer.
        ffn: Feed-forward sublayer.
    """

    def __init__(self, config: DecoderLMConfig):
        super().__init__()
        self.use_residual = config.use_residual
        self.attn_norm = _build_norm(config.d_model, config.norm_type)
        self.ffn_norm = _build_norm(config.d_model, config.norm_type)
        self.attn = CausalSelfAttention(config)
        self.ffn = FeedForward(config)

    def _merge_residual(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        """Merge a sublayer update with its residual stream.

        Args:
            residual: Input tensor before the sublayer.
            update: Output tensor produced by the sublayer.

        Returns:
            The residual sum when enabled, otherwise the raw update tensor.
        """
        if self.use_residual:
            return residual + update
        return update

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and FFN sublayers in sequence.

        Args:
            x: Input tensor with shape ``[batch, seq_len, d_model]``.

        Returns:
            Tensor with the same shape as ``x`` after one decoder block.
        """
        x = self._merge_residual(x, self.attn(self.attn_norm(x)))
        x = self._merge_residual(x, self.ffn(self.ffn_norm(x)))
        return x


class DecoderOnlyTransformerLM(nn.Module):
    """Compact decoder-only Transformer language model.

    The model maps token ids to next-token logits through token embeddings,
    stacked decoder blocks, a final normalization layer, and a vocabulary-sized
    output projection.

    Attributes:
        config: Validated model hyper-parameters.
        token_embedding: Embedding table for discrete token ids.
        embedding_dropout: Dropout applied after token embedding lookup.
        blocks: Ordered stack of decoder blocks.
        final_norm: Optional normalization before logits projection.
        lm_head: Output projection from hidden states to vocabulary logits.
    """

    def __init__(self, config: DecoderLMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = _build_norm(config.d_model, config.norm_type)
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=config.bias,
        )

        if config.tie_embeddings:
            # Share the token embedding matrix with the output projection.
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Compute per-token logits for a batch of token id sequences.

        Args:
            input_ids: Tensor of token ids with shape ``[batch, seq_len]``.

        Returns:
            Logits with shape ``[batch, seq_len, vocab_size]``.

        Raises:
            ValueError: If ``input_ids`` does not have rank 2.
            ValueError: If the sequence length exceeds ``config.max_seq_len``.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")

        seq_len = input_ids.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                "input sequence length exceeds config.max_seq_len"
            )

        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.embedding_dropout(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)
