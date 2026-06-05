import numpy as np
from tokenizer.tokenizer import tokenizer
from config_manager import load_BPEConfig, load_DataConfig

def main():
    bpeconfig = load_BPEConfig()
    dataconfig = load_DataConfig()

    tokeni = tokenizer.from_files(
        bpeconfig.vocab_path,
        bpeconfig.merge_path,
        bpeconfig.special_tokens,
    )

    with open(dataconfig.TinyStories_train_path, "r", encoding="utf-8") as in_f:
        with open(dataconfig.training_token_path, "wb") as out_f:
            token_stream = tokeni.encode_iterable(in_f)
            
            chunk = []
            
            for token in token_stream:
                chunk.append(token)
                if len(chunk) >= 1_000_000:  
                    np.array(chunk, dtype=np.uint16).tofile(out_f)
                    chunk.clear()
            
            
            if chunk:
                np.array(chunk, dtype=np.uint16).tofile(out_f)


if __name__ == "__main__":
    main()