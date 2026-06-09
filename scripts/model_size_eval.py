from config_manager import load_ModelConfig

def main():
    cfg= load_ModelConfig()
    print(f"Model config: {cfg}")

    embeddings_size = cfg.d_model * cfg.vocab_size
    print(f"Embedding size (number of parameters): {embeddings_size:,}")

    attention_size = 4 * cfg.d_model * cfg.d_model 
    print(f"Attention size per layer (number of parameters): {attention_size:,}")

    ffn_size = 3 * cfg.d_model * cfg.d_ff
    print(f"FFN layers size  per layer (number of parameters): {ffn_size:,}")

    transformer_size = (attention_size + ffn_size) * cfg.num_layers
    print(f"TransformerLM size (number of parameters): {transformer_size:,}")    

    output_size = cfg.d_model * cfg.vocab_size
    print(f"Output layer size (number of parameters): {output_size:,}")

    total_size = embeddings_size + transformer_size + output_size
    print(f"Total model size (number of parameters): {total_size:,}")

    size_in_mb = total_size * 4 / (1024 ** 2)  # Assuming 4 bytes per parameter (float32)
    print(f"Total model size: {size_in_mb:.2f} MB")
    return 


if __name__ == "__main__":
    main()
