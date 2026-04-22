import yaml
from dataclasses import dataclass


config_path: str = "config/default.yaml"

@dataclass
class BPEConfig:
    num_process: int
    vocab_size: int
    special_tokens: list[str]
    vocab_path: str
    merge_path: str


@dataclass
class DataConfig:
    owt_train_path: str
    owt_valid_path: str
    TinyStories_train_path: str
    TinyStories_valid_path: str


VALID_FFN_TYPES = {"swiglu", "gelu", "silu"}
VALID_NORM_TYPES = {"rmsnorm", "layernorm", "none"}


@dataclass(slots=True)
class DecoderLMConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    num_layers: int
    num_heads: int
    ffn_hidden_dim: int
    norm_type: str
    ffn_type: str
    use_residual: bool
    dropout: float
    tie_embeddings: bool
    rope_base: float
    bias: bool

    def __post_init__(self) -> None:
        self.norm_type = self.norm_type.lower()
        self.ffn_type = self.ffn_type.lower()

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.ffn_hidden_dim <= 0:
            raise ValueError("ffn_hidden_dim must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if (self.d_model // self.num_heads) % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0 inclusive")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")
        if self.norm_type not in VALID_NORM_TYPES:
            valid_norm_types = ", ".join(sorted(VALID_NORM_TYPES))
            raise ValueError(
                f"norm_type must be one of: {valid_norm_types}"
            )
        if self.ffn_type not in VALID_FFN_TYPES:
            valid_ffn_types = ", ".join(sorted(VALID_FFN_TYPES))
            raise ValueError(
                f"ffn_type must be one of: {valid_ffn_types}"
            )



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


def _load_config_data() -> dict:
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def load_BPEConfig() -> BPEConfig:
    config_data = _load_config_data()
    return BPEConfig(**config_data["bpe"])


def load_DataConfig() -> DataConfig:
    config_data = _load_config_data()
    return DataConfig(**config_data["data"])


def load_DecoderLMConfig() -> DecoderLMConfig:
    config_data = _load_config_data()
    return DecoderLMConfig(**config_data["model"])


def load_TrainingConfig() -> TrainingConfig:
    config_data = _load_config_data()
    return TrainingConfig(**config_data["training"])
