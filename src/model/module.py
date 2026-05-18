""" Basic module class for language models.

"""

import torch
import torch.nn as nn
from einops import rearrange

class Linear(nn.Module):
    """
    My linear moudle that mimics torch.nn.linear.

    Attributes:
        weight: The weight parameter of the linear module.
    """
    def __init__(
        self,
        in_features: int, 
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ): 
        super().__init__()
        self.weight = nn.Parameter(nn.init.trunc_normal_(
            torch.empty(out_features, in_features, dtype=dtype, device=device)))

        
    def forward(
        self,
        x : torch.Tensor
    ) -> torch.Tensor:
        w_trans = rearrange(self.weight, 'd_out d_in -> d_in d_out')
        return x @ w_trans
    
class Embedding(nn.Module):
    def __init__(
        self,
        num_embedings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(nn.init.trunc_normal_(
            torch.empty(num_embedings, embedding_dim, dtype=dtype, device=device)))
        
    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.weight[token_ids]