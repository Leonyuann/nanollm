from dataclasses import dataclass


VALID_FFN_TYPES = {"swiglu", "gelu", "silu"}
VALID_NORM_TYPES = {"rmsnorm", "layernorm", "none"}


@dataclass(slots=True)
class DecoderLMConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    num_layers: int
    num_heads: int
    ffn_hidden_dim: int
    norm_type: str
    ffn_type: str
    use_residual: bool
    dropout: float
    tie_embeddings: bool
    rope_base: float
    bias: bool

    def __post_init__(self) -> None:
        self.norm_type = self.norm_type.lower()
        self.ffn_type = self.ffn_type.lower()

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.ffn_hidden_dim <= 0:
            raise ValueError("ffn_hidden_dim must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if (self.d_model // self.num_heads) % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0 inclusive")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")
        if self.norm_type not in VALID_NORM_TYPES:
            valid_norm_types = ", ".join(sorted(VALID_NORM_TYPES))
            raise ValueError(
                f"norm_type must be one of: {valid_norm_types}"
            )
        if self.ffn_type not in VALID_FFN_TYPES:
            valid_ffn_types = ", ".join(sorted(VALID_FFN_TYPES))
            raise ValueError(
                f"ffn_type must be one of: {valid_ffn_types}"
            )

