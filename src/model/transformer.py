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
        rms = torch.sqrt(einsum(x, x, "... d, ... d -> ...")/self.d_model + self.eps)
        return self.gain * (x / rearrange(rms, '... -> ... 1'))
        
        
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