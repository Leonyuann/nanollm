import random
import torch
from tqdm import tqdm
import argparse
import numpy as np
from loguru import logger
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import time


from model import transformer
from training import loss, optimizer, data, checkpoint
from config_manager import load_ModelConfig, load_AdamWConfig, load_DataConfig, load_TrainingConfig
from logger import WandbLogger, local_record
from model_size_eval import model_size_in_mb

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
    parser.add_argument("--device", type=torch.device, default=model_config.device)
    parser.add_argument(
        "--dtype",
        type=lambda value: getattr(torch, value),
        default=model_config.dtype,
    )

    # Optimizer config
    parser.add_argument("--lr", type=float, default=optimizer_config.lr)
    parser.add_argument("--beta1", type=float, default=optimizer_config.beta1)
    parser.add_argument("--beta2", type=float, default=optimizer_config.beta2)
    parser.add_argument("--eps", type=float, default=optimizer_config.eps)
    parser.add_argument("--weight_decay", type=float, default=optimizer_config.weight_decay)
    parser.add_argument("--min_lr_ratio", type=float, default=optimizer_config.min_lr_ratio)
    parser.add_argument("--warmup_steps", type=int, default=optimizer_config.warmup_steps)
    parser.add_argument("--decay_steps", type=int, default=optimizer_config.decay_steps)


    # Data config
    parser.add_argument("--training_data_path", type=str, default=data_config.training_data_path)
    parser.add_argument("--eval_data_path", type=str, default=data_config.evaluation_data_path)

    # Training config
    parser.add_argument("--seed", type=int, default=training_config.seed)
    parser.add_argument("--training_step", type=int, default=training_config.training_step)
    parser.add_argument("--eval_step", type=int, default=training_config.eval_step)
    parser.add_argument("--eval_every", type=int, default=training_config.eval_every)
    parser.add_argument("--save_every", type=int, default=training_config.save_every)
    parser.add_argument("--save_dir", type=str, default=training_config.save_dir)
    parser.add_argument("--batch_size", type=int, default=training_config.batch_size)
    parser.add_argument("--eval_batch_size", type=int, default=training_config.eval_batch_size)
    parser.add_argument(
        "--use_wandb",
        action=argparse.BooleanOptionalAction,
        default=training_config.use_wandb,
    )
    parser.add_argument("--use_log",
        action=argparse.BooleanOptionalAction,
        default=training_config.use_log,
    )
    parser.add_argument("--log_path", type=str, default=training_config.log_path)
    parser.add_argument("--max_gradient_norm", type=float, default=training_config.max_gradient_norm)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed all random number generators used during training.

    Args:
        seed: Non-negative random seed.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)

    model = transformer.TransformerLM(
        vocab_size= args.vocab_size,
        context_length= args.context_length,
        num_layers= args.num_layers,
        d_model= args.d_model,
        num_heads= args.num_heads,
        d_ff= args.d_ff,
        rope_theta= args.rope_theta,
        device=args.device,
        dtype=args.dtype,
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

    loop(args, model, optim)


def eval_model (
    args: argparse.Namespace,
    model: transformer.TransformerLM,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    eval_data = np.memmap(args.eval_data_path, dtype=np.uint16, mode='r')

    mean_loss = 0.0
    for i in range(args.eval_step):
        with torch.no_grad():
            sample ,target= data.data_loading(eval_data, args.eval_batch_size, args.context_length, args.device)

            logits = model(sample)
            celoss = loss.cross_entropy(logits, target)

            mean_loss -= 1 / (i + 1) * (mean_loss - celoss)

    ppl = torch.exp(mean_loss)
    model.train()
    return mean_loss, ppl


def save_checkpoint_to_dir(
    model: transformer.TransformerLM,
    optim: torch.optim.Optimizer,
    step: int,
    save_dir: str,
):
    size_in_mb = model_size_in_mb(model)
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%m%d-%H%M")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    file_name = save_path / f"LM-{size_in_mb:.1f}MB-{timestamp}-{step}.pt"

    checkpoint.save_checkpoint(model, optim, step, file_name)
    logger.info(f"Checkpoint saved at step {step} to {file_name}")



def loop (
    args: argparse.Namespace,
    model: transformer.TransformerLM,
    optim: torch.optim.Optimizer,
):
    wblogger = WandbLogger(args)
    # Semantics: finished training steps number
    global_step = 0

    # Lazy-load data
    training_data = np.memmap(args.training_data_path, dtype=np.uint16, mode='r')

    # Progress bar
    pbar = tqdm(total=args.training_step, desc="Training", unit="step")

    # Time record
    start_time = time.time()
    # Traing loop
    while global_step < args.training_step:
        sample, target = data.data_loading(training_data, args.batch_size, args.context_length, args.device)

        logits = model(sample)
        celoss = loss.cross_entropy(logits, target)
        celoss.backward()

        # Apply gradient clipping
        optimizer.gradient_clipping(model.parameters(),args.max_gradient_norm)

        # Apply cosine learning rate schedule
        lr = optimizer.cosine_learning_lr_schedule(
            step=global_step + 1,
            max_lr=args.lr,
            min_lr=args.min_lr_ratio * args.lr,
            t_w=args.warmup_steps,
            t_c=args.decay_steps,
        )

        for group in optim.param_groups:
            group["lr"] = lr

        # Update parameters and zero gradients
        optim.step()
        optim.zero_grad()

        # Update global step
        global_step += 1

        # Update progress bar and log training metrics
        pbar.update(1)
        wblogger.train_log(
            {
                "loss": celoss.item(),
                "lr": optim.param_groups[0]['lr'],
            },
            step=global_step
        )

        # Evaluation
        if args.eval_every > 0 and global_step % args.eval_every == 0:
            eval_loss, ppl = eval_model(args, model)
            wblogger.eval_log(
                {
                    "loss": eval_loss.item(),
                    "ppl": ppl.item(),
                },
                step=global_step
            )

        # Checkpointing
        if args.save_every > 0 and global_step % args.save_every == 0 and global_step != args.training_step:
            save_checkpoint_to_dir(model, optim, global_step, args.save_dir)
    # End of training loop
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Final checkpoint and record
    save_checkpoint_to_dir(model, optim, global_step, args.save_dir)

    # if use_log is enabled, save the training record to local file
    if args.use_log:
        local_record(
            args.log_path,
            {
                "run-id": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%m%d-%H%M"),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "data": args.context_length * args.batch_size * args.training_step,
                "compute_budget": 6 * sum(parameter.numel() for parameter in model.parameters()) * args.context_length * args.batch_size * args.training_step,
                "num_layers": args.num_layers,
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "sequence_length": args.context_length,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "weight_decay": args.weight_decay,
                "wall_clock_seconds": elapsed_time,
                "valid_loss": eval_loss.item(),
            }
        )
    pbar.close()
    wblogger.finish()

if __name__ == "__main__":
    logger.info("TRAINING START")
    args = parse_args()
    train(args)
