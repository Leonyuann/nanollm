from config_manager import load_ModelConfig
from model import transformer


def model_size_in_mb(model: transformer.TransformerLM) -> float:
    size_in_bytes = sum(
    parameter.numel() * parameter.element_size()
    for parameter in model.parameters()
    )
    return size_in_bytes / 1024**2

def main():
    cfg= load_ModelConfig("config/42,53.yaml")
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

    model = transformer.TransformerLM(
        vocab_size= cfg.vocab_size,
        context_length= cfg.context_length,
        num_layers= cfg.num_layers,
        d_model= cfg.d_model,
        num_heads= cfg.num_heads,
        d_ff= cfg.d_ff,
        rope_theta= cfg.rope_theta,
        device= cfg.device,
        dtype= cfg.dtype,
    )
    size_in_mb = model_size_in_mb(model)
    print(f"Total model size: {size_in_mb:.2f} MB")
    return 


if __name__ == "__main__":
    main()
