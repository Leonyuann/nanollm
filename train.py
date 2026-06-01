import os

import torch
import argparse

from loguru import logger
from model import module, transformer
from training import loss, optimizer, data, checkpoint
from config_manager import load_ModelConfig, load_AdamWConfig


def parse_args() -> argparse.Namespace:
    model_config = load_ModelConfig()
    optimizer_config = load_AdamWConfig()

    # Model config
    parser = argparse.ArgumentParser(description="nanoLLM training loop.")
    parser.add_argument("--vocab_size" ,type=int, default=model_config.vocab_size)
    parser.add_argument("--context_length",type=int,default=model_config.context_length)
    parser.add_argument("--num_layers", type=int, default=model_config.num_layers)
    parser.add_argument("--d_model", type=int, default=model_config.d_model)
    parser.add_argument("--num_heads",type=int, default=model_config.num_heads)
    parser.add_argument("--d_ff", type=int, default=model_config.d_ff)
    parser.add_argument("--rope_theta", type=float, default=model_config.rope_theta)
    parser.add_argument("--device", default=model_config.device)
    parser.add_argument("--dtype", default=model_config.dtype)


    # Optimizer config
    parser.add_argument("--lr", type=float, default=optimizer_config.lr)
    parser.add_argument("--beta1", type=float, default=optimizer_config.beta1)
    parser.add_argument("--beta2", type=float, default=optimizer_config.beta2)
    parser.add_argument("--eps", type=float, default=optimizer_config.eps)
    parser.add_argument("--weight_decay", type=float, default=optimizer_config.weight_decay)
    
    return parser.parse_args()

def train(args):

    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    model = transformer.TransformerLM(
        vocab_size= args.vocab_size,
        context_length= args.context_length,
        num_layers= args.num_layers,
        d_model= args.d_model,
        num_heads= args.num_heads,
        d_ff= args.d_ff,
        rope_theta= args.rope_theta,
        device= args.device,
        dtype= args.dtype,
    )
    logger.info("Model initialized.")

    optim = optimizer.AdamW(
        model.parameters(),
        lr= args.lr,
        betas= (args.beta1, args.beta2),
        eps= args.eps,
        weight_decay= args.weight_decay,
    )
    logger.info("Optimizer initialized.")

if __name__ == "__main__":
    print ("TRAINING START")
    args = parse_args()
    train(args)
