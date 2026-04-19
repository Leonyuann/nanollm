import yaml
from dataclasses import dataclass

@dataclass
class BPEConfig:
    num_process: int
    vocab_size: int
    special_tokens: list[str]
    vocab_path: str
    merge_path: str


def load_BPEConfig(config_path: str) -> BPEConfig:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)
    
    return BPEConfig(**config_data["bpe"])
