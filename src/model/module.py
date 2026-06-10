""" Basic module class for language models.

This module implement personal basic moudle to substitute torch.nn.Linear 
and torch.nn.Embedding.

Class:
    Linear: A linear module that mimics torch.nn.Linear, but only has weight and no bias.
    Embedding: An embedding module that mimics torch.nn.Embedding.
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
            torch.empty(out_features, in_features, dtype=dtype, device=device),
            std=2/(in_features + out_features),
            a=-3 * (2/(in_features + out_features))**0.5,
            b=3 * (2/(in_features + out_features))**0.5
        ))

        
    def forward(
        self,
        x : torch.Tensor
    ) -> torch.Tensor:
        w_trans = rearrange(self.weight, 'd_out d_in -> d_in d_out')
        return x @ w_trans
    
class Embedding(nn.Module):
    """
    My embedding moudle that mimics torch.nn.Embedding.

    Attributes:
        weight: The weight parameter of the embedding module, 
        which contains the embeddings for each token ID.
    """
    def __init__(
        self,
        num_embedings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(nn.init.trunc_normal_(
            torch.empty(num_embedings, embedding_dim, dtype=dtype, device=device),
            std=1.0,
            a=-3.0,
            b=3.0,
        ))
        
    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.weight[token_ids]