from config_manager import TrainingConfig


__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "TokenSequenceIterableDataset",
    "WarmupConstantScheduler",
    "compute_next_token_loss",
    "run_training",
]


def __getattr__(name: str):
    """Lazily expose training helpers while avoiding import cycles.

    Args:
        name: Requested attribute name.

    Returns:
        The exported training symbol.

    Raises:
        AttributeError: If ``name`` is not an exported symbol.
    """
    if name == "TokenSequenceIterableDataset":
        from .data import TokenSequenceIterableDataset

        return TokenSequenceIterableDataset
    if name in {
        "TrainingResult",
        "WarmupConstantScheduler",
        "compute_next_token_loss",
        "run_training",
    }:
        from .train import (
            TrainingResult,
            WarmupConstantScheduler,
            compute_next_token_loss,
            run_training,
        )

        namespace = {
            "TrainingResult": TrainingResult,
            "WarmupConstantScheduler": WarmupConstantScheduler,
            "compute_next_token_loss": compute_next_token_loss,
            "run_training": run_training,
        }
        return namespace[name]
    raise AttributeError(f"module 'training' has no attribute {name!r}")
