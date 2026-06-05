import os

from tokenizer import train_bpe
from config_manager import load_BPEConfig

bpeconfig = load_BPEConfig()

vocab , merges = train_bpe.train_bpe(
    input_path="/xts001/owt_train.txt",
    vocab_size=bpeconfig.vocab_size,
    special_tokens=bpeconfig.special_tokens,
)

os.makedirs(os.path.dirname(bpeconfig.merge_path), exist_ok=True)
os.makedirs(os.path.dirname(bpeconfig.vocab_path), exist_ok=True)

with open(bpeconfig.vocab_path, "w", encoding="utf-8") as f:
    for key in vocab:
        f.write(f"{key}\t{vocab[key].hex()}\n")

with open(bpeconfig.merge_path, "w", encoding="utf-8") as f:
    for bytes1, bytes2 in merges:
        f.write(f"{bytes1.hex()}\t{bytes2.hex()}\n")


