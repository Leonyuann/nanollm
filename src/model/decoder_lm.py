import torch
from torch import nn
from torch.nn import functional as F

from .config import DecoderLMConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(mean_square + self.eps)
        return normalized * self.weight


def _build_norm(dim: int, norm_type: str) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {norm_type}")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
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
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)
        cos = cos.unsqueeze(0).unsqueeze(0).to(dtype)
        sin = sin.unsqueeze(0).unsqueeze(0).to(dtype)
        return cos, sin


class FeedForward(nn.Module):
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
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

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
        if self.use_residual:
            return residual + update
        return update

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._merge_residual(x, self.attn(self.attn_norm(x)))
        x = self._merge_residual(x, self.ffn(self.ffn_norm(x)))
        return x


class DecoderOnlyTransformerLM(nn.Module):
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
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
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
