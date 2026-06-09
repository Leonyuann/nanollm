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
    training_data_path: str
    evaluation_data_path: str

@dataclass
class TransformerConfig:
    RMSNorm_eps: float

@dataclass
class AdamWConfig:
    lr: float
    beta1: float
    beta2: float
    weight_decay: float
    eps: float

@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: float
    device: str | None
    dtype: str | None

@dataclass
class TrainingConfig:
    seed: int
    training_step: int
    eval_step: int
    eval_every: int
    save_every: int
    save_dir: str
    batch_size: int
    eval_batch_size: int
    use_wandb: bool

def load_BPEConfig() -> BPEConfig:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)
    
    return BPEConfig(**config_data["bpe"])

def load_DataConfig() -> DataConfig:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)
    
    return DataConfig(**config_data["data"])

def load_TransformerConfig() -> TransformerConfig:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)
    
    return TransformerConfig(**config_data["transformer"])

def load_AdamWConfig() -> AdamWConfig:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)

    return AdamWConfig(**config_data["adamw"])

def load_ModelConfig() -> ModelConfig:
    """Load model hyperparameters from the default YAML config.

    Returns:
        ModelConfig: Model configuration with PyTorch device and dtype values.
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)

    return ModelConfig(**config_data["model"])

def load_TrainingConfig() -> TrainingConfig:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)
    
    return TrainingConfig(**config_data["training"])