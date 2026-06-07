"""Loss functions for training language models."""

import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange

def cross_entropy(
    logits: Float[Tensor,"... seq_len vocab_size"],
    target: Int[Tensor, "... seq_len"]
) -> Float[Tensor,""]:
    """
    Args:
        logits: Unnormalized class scores with vocabulary size on the final
            dimension.
        target: Index of the correct class for each example. Its shape must
            match `logits.shape[:-1]`.

    Returns:
        Average negative log likelihood across all target positions.

    Raises:
        ValueError: If `logits` has no class dimension or `target` does not
            match the leading dimensions of `logits`.
    """
    if logits.ndim == 0:
        raise ValueError("logits must include a class dimension.")
    if logits.shape[:-1] != target.shape:
        raise ValueError(
            f"target shape {tuple(target.shape)} must match logits leading "
            f"shape {tuple(logits.shape[:-1])}."
        )
    logits_max = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - logits_max

    sum_log_exp = torch.logsumexp(shifted_logits, dim=-1)
    target_logits = torch.gather(
        shifted_logits, 
        dim = -1,
        index=rearrange(target, "... -> ... 1")
    )

    target_logits = rearrange(target_logits, "... 1 -> ...")

    return torch.mean(sum_log_exp - target_logits)

def perplexity(
    logits: Float[Tensor,"... seq_len vocab_size"],
    target: Int[Tensor, "... seq_len"]
) -> Float[Tensor,""]:
    """
    Args:
        logits: Unnormalized class scores with vocabulary size on the final
            dimension.
        target: Index of the correct class for each example. Its shape must
            match `logits.shape[:-1]`.
    Returns:
        Exponential of the average negative log likelihood across all target
        positions.
    Raises:
        ValueError: If `logits` has no class dimension or `target` does not
            match the leading dimensions of `logits`.

    """
    return torch.exp(cross_entropy(logits, target))
