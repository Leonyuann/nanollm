"""Data loading and batching for training.
"""
import numpy as np
import numpy.typing as npt
import torch


def data_loading(
    x: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random language-modeling batches from a token array.

    Args:
        x: One-dimensional numpy array of token IDs.
        batch_size: Number of sequences to sample.
        context_length: Number of input tokens per sequence.
        device: Target PyTorch device, such as "cpu" or "cuda:0".

    Returns:
        Input and target tensors with shape (batch_size, context_length).

    Raises:
        ValueError: If arguments are invalid or the dataset is too short.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional token array.")
    if len(x) < context_length + batch_size:
        raise ValueError("dataset is too short to sample one sequence.")

    starts = torch.randint(
        low = 0,
        high = len(x) - context_length,
        size = (batch_size,),
    )

    sample_stacks = np.stack([x[i: i + context_length + 1] for i in starts.numpy()])
    sample_tensors = torch.from_numpy(sample_stacks.astype(np.int64)).to(device)

    sequences = sample_tensors[:,:-1]
    targets = sample_tensors[:, 1:]

    return sequences, targets