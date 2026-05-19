""" Implementation of the Transformer architecture
"""

import torch
from einops import einsum, rearrange

class RMSNorm(torch.nn.Module):
    """
    Implements Root Mean Square Normalization.

    Attributes:
        eps: A small value added to the denominator for numerical stability.
        d_model: The dimension of the input features.
        gain: A learnable parameter that scales the normalized output, initilized to 1  .
    """
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
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
        rms = torch.sqrt(einsum(x, x, "... d, ... d -> ...")/self.d_model + self.eps)
        return self.gain * (x / rearrange(rms, '... -> ... 1'))
        
        

    