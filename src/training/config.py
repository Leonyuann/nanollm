from dataclasses import dataclass


VALID_DATASETS = {"tinystories", "owt"}
VALID_DEVICES = {"auto", "cpu", "cuda"}


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for the language-model training loop.

    Extended description.

    Attributes:
        dataset: Dataset alias used to resolve train and validation files.
        batch_size: Number of token windows per optimization step.
        max_steps: Total number of optimizer steps to run.
        learning_rate: Base AdamW learning rate after warmup finishes.
        warmup_steps: Number of linear warmup steps before the constant phase.
        log_interval: Step interval for training-loss logging.
        eval_interval: Step interval for validation evaluation.
        eval_steps: Maximum number of validation batches per evaluation run.
        device: Device selection mode.
        seed: Random seed used for PyTorch.

    Notes:
        The sequence length is inherited from the model config.
    """

    dataset: str
    batch_size: int
    max_steps: int
    learning_rate: float
    warmup_steps: int
    log_interval: int
    eval_interval: int
    eval_steps: int
    device: str
    seed: int

    def __post_init__(self) -> None:
        """Validate and normalize training-loop configuration values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If any config value is outside the supported range.
        """
        self.dataset = self.dataset.lower()
        self.device = self.device.lower()

        if self.dataset not in VALID_DATASETS:
            valid_datasets = ", ".join(sorted(VALID_DATASETS))
            raise ValueError(f"dataset must be one of: {valid_datasets}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive")
        if self.device not in VALID_DEVICES:
            valid_devices = ", ".join(sorted(VALID_DEVICES))
            raise ValueError(f"device must be one of: {valid_devices}")
