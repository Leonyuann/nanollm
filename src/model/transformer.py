""" Implementation of the Transformer architecture
"""

import torch
from einops import einsum, rearrange

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
        self.up_project = module.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.down_project = module.Linear(d_ff, d_model, device=device, dtype=dtype)
        

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        gate = self.gate(x)
        gated_score = gate * torch.sigmoid(gate)
        up = gated_score * self.up_project(x)
        down = self.down_project(up)
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