import yaml
from dataclasses import dataclass

from model.config import DecoderLMConfig

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
