import os

import torch
import argparse
import wandb 
import numpy as np

from loguru import logger
from model import module, transformer
from training import loss, optimizer, data, checkpoint
from config_manager import load_ModelConfig, load_AdamWConfig, load_DataConfig, load_TrainingConfig


def parse_args() -> argparse.Namespace:
    model_config = load_ModelConfig()
    optimizer_config = load_AdamWConfig()
    data_config = load_DataConfig()
    training_config = load_TrainingConfig()

    parser = argparse.ArgumentParser(description="nanoLLM training loop.")

    # Model config
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

    # Data config
    parser.add_argument("--training_data_path", type=str, default=data_config.TinyStories_train_path)
    parser.add_argument("--eval_data_path", type=str, default=data_config.TinyStories_valid_path)

    # Training config
    parser.add_argument("--total_step", type=int, default=training_config.training_step)
    parser.add_argument("--eval_every", type=int, default=training_config.eval_every)
    parser.add_argument("--save_every", type=int, default=training_config.save_every)
    parser.add_argument("--save_dir", type=str, default=training_config.save_dir)
    parser.add_argument("--batch_size", type=str, default=training_config.batch_size)
    
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

    # Semantics: finished training steps number
    global_step = 0

    # Lazy-load data
    training_data = np.memmap(args.training_data_path, dtype=int, mode='r')
    eval_data = np.memmap(args.eval_data_path, dtype=int, mpde='r')

    while global_step < args.total_step:
        batch = data.data_loading(training_data, args.batch_size, args.context_length, args.device)
        loss = model(batch)

        loss.backward()

        optim.step()
        optim.zero_grad()

        global_step += 1

    logger.info("*" * 40)
    logger.info("Training finish.")
    logger.info("*" * 40)

if __name__ == "__main__":
    logger.info("TRAINING START")
    args = parse_args()
    train(args)
