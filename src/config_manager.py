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

@dataclass
class TransformerConfig:
    RMSNorm_eps: float

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

