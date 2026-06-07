""" Implementation of the Transformer architecture
"""

import torch
import math
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int

from config_manager import load_TransformerConfig
from model import module

transformer_config = load_TransformerConfig()
RMSNORM_EPS = transformer_config.RMSNorm_eps

class RMSNorm(torch.nn.Module):
    """
    Implements Root Mean Square Normalization.

    Attributes:
        eps: A small value added to the denominator for numerical stability.
        d_model: The dimension of the input features.
        gain: A learnable parameter that scales the normalized output, initilized to 1.
    """
    def __init__(
        self,
        d_model: int,
        eps: float = RMSNORM_EPS,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ): 
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.gain = torch.nn.Parameter(
            torch.ones(d_model, dtype = dtype, device = device)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(einsum(x, x, "... d, ... d -> ...")/self.d_model + self.eps)
        result = self.gain * (x / rearrange(rms, '... -> ... 1'))

        # Return the result in the original dtype
        return result.to(in_dtype)
        
        
class SwiGLU(torch.nn.Module):
    """
    Implements the SwiGLU activation function.
    
    Attributes:
        gate: A linear layer that projects the input to 
            a higher dimension for the gating mechanism.
        up_project: A linear layer that projects the input to 
            a higher dimension for the up projection
        down_project: A linear layer that projects the output of the 
            up projection back to the original dimension for the down projection.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # TODO: make sure d_swiglu is an integer 
        # and a nearby multiple of 64 for hardware efficency
        self.gate = module.Linear(d_model, d_ff, device=device, dtype=dtype) 
        self.up_proj = module.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.down_proj = module.Linear(d_ff, d_model, device=device, dtype=dtype)
        

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        gate = self.gate(x)
        gated_score = gate * torch.sigmoid(gate)
        up = gated_score * self.up_proj(x)
        down = self.down_proj(up)
        return down
    

class RotaryPositionalEmbedding(torch.nn.Module):
    """
    Implements Rotary Positional Embedding (RoPE) as described in the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding".

    Attributes:
        theta: A base frequency for the rotary embeddings.
        d_k: The dimension of the input features that will be rotated.
        max_seq_len: The maximum sequence length for which the rotary embeddings will be precomputed.
        cos_cached: A buffer that stores the precomputed cosine values for the rotary embeddings.
        sin_cached: A buffer that stores the precomputed sine values for the rotary embeddings.
    """
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ): 
        super().__init__()

        assert d_k % 2 == 0, "d_k must be even for RoPE."

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Compute the inverse frequencies for the rotary embeddings
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        positions = torch.arange(max_seq_len, device=device)
        freqs = einsum(positions, inv_freq, "n, d -> n d")

        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        
        assert x.shape[-1] == self.d_k, (
            f"Expected x.shape[-1] == {self.d_k}, got {x.shape[-1]}"
        )

        assert token_positions.max() < self.max_seq_len, (
            f"token_positions exceed max_seq_len={self.max_seq_len}"
        )

        # 
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x_routed = torch.empty_like(x)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_routed[..., 0::2] = cos * x_even - sin * x_odd
        x_routed[..., 1::2] = sin * x_even + cos * x_odd
        
        return x_routed
    
def softmax(
    x : torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """
    Softmax implementation.
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum


def scaled_dot_prodoct_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... keys d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    """
    Scaled dot-product attention implementation.
    """
    scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k")
    scores = scores / math.sqrt(Q.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    
    attn_weights = softmax(scores, dim=-1)
    attn = einsum(attn_weights, V, "... q k, ... k d_v -> ... q d_v")
    return attn

class MultiHeadSelfAttention(torch.nn.Module):
    """
    Implements multi-head self-attention with RoPE.

    Attributes:
        d_model: The dimension of the input features.
        num_heads: The number of attention heads.
        d_head: The dimension of each attention head, calculated as d_model // num_heads.
        rope: An instance of the RotaryPositionalEmbedding class for applying RoPE to the query
            and key tensors.
        qkv_proj: A linear layer that projects the input features to the query, key,
        
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by nums_heads."
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.device = device

        self.rope = RotaryPositionalEmbedding(
            theta = theta, 
            d_k = self.d_head, 
            max_seq_len = max_seq_len,
            device = device,
        )
        self.qkv_proj = module.Linear(d_model, 3 * d_model, device, dtype)
        self.o_proj = module.Linear(d_model, d_model, device, dtype)

    def multi_head(
        self,
        Q: Float[torch.Tensor, " ... queries d_model"],
        K: Float[torch.Tensor, " ... keys d_model"],
        V: Float[torch.Tensor, " ... keys d_model"],
    ) -> Float[torch.Tensor, " ... queries d_model"]:
        Q = rearrange(Q, "... q (h d) -> ... h q d", h = self.num_heads, d = self.d_head)
        K = rearrange(K, "... k (h d) -> ... h k d", h = self.num_heads, d = self.d_head)
        V = rearrange(V, "... v (h d) -> ... h v d", h = self.num_heads, d = self.d_head)

        q_seq_len = Q.size(-2)
        v_seq_len = V.size(-2)
        # Generate token positions for Q and K
        q_token_positions = torch.arange(q_seq_len)
        k_token_positions = torch.arange(v_seq_len)

        # Apply RoPE to Q and K
        Q_rope = self.rope(Q, q_token_positions)
        K_rope = self.rope(K, k_token_positions)

        mask = torch.full((q_seq_len, v_seq_len), fill_value=True, dtype=torch.bool, device=self.device)
        mask = ~torch.triu(mask, diagonal=1)

        atten = scaled_dot_prodoct_attention(Q_rope, K_rope, V, mask)
        atten = rearrange(atten, "... h q d -> ... q (h d)", h = self.num_heads, d = self.d_head)

        return atten
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        assert x.shape[-1] == self.d_model, (
            f"Expected x.shape[-1] == {self.d_model}, got {x.shape[-1]}"
        )

        assert x.size(-2) <= self.rope.max_seq_len, (
            f"Sequence length {x.size(-2)} exceeds max_seq_len={self.rope.max_seq_len}"
        )
        QKV = self.qkv_proj(x)
        Q, K, V = rearrange(QKV, "... (k d_model) -> k ... d_model", k=3,d_model=self.d_model)

        O = self.multi_head(Q, K, V)

        return self.o_proj(O)

class TransformerBlock(torch.nn.Module):
    """
    Implements a single block of the Transformer architecture.

    Attributes:
        ln1: normalization layer applied before the self-attention mechanism.
        attn: An instance of the MultiHeadSelfAttention class that implements the self-attention
            mechanism.
        ln2: normalization layer applied before the feed-forward network.
        ffn: An instance of the SwiGLU class that implements the feed-forward network.
    

    """

    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model = d_model,
            num_heads = num_heads,
            theta = theta, 
            max_seq_len = max_seq_len,
            device = device,
            dtype = dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device ,dtype)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class TransformerLM(torch.nn.Module):
    """
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ): 
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model

        self.token_embeddings = module.Embedding(vocab_size, d_model, device ,dtype)
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(
                d_model = d_model, 
                num_heads = num_heads, 
                theta = rope_theta,
                max_seq_len = context_length, 
                d_ff = d_ff,
                device = device,
                dtype = dtype,
            ) for i in range(num_layers)]
        )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = module.Linear(d_model, vocab_size, device , dtype)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.token_embeddings(x)

        for transformer_block in self.layers:
            hidden_states = transformer_block(hidden_states)
        
        hidden_states = self.ln_final(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

