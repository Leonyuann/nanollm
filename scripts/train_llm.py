import torch
from tqdm import tqdm
import argparse
import numpy as np
from loguru import logger

from model import transformer
from training import loss, optimizer, data, checkpoint
from config_manager import load_ModelConfig, load_AdamWConfig, load_DataConfig, load_TrainingConfig
from logger import WandbLogger

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

    # Data config
    parser.add_argument("--training_data_path", type=str, default=data_config.training_data_path)
    parser.add_argument("--eval_data_path", type=str, default=data_config.evaluation_data_path)

    # Training config
    parser.add_argument("--total_step", type=int, default=training_config.training_step)
    parser.add_argument("--eval_every", type=int, default=training_config.eval_every)
    parser.add_argument("--save_every", type=int, default=training_config.save_every)
    parser.add_argument("--save_dir", type=str, default=training_config.save_dir)
    parser.add_argument("--batch_size", type=int, default=training_config.batch_size)
    parser.add_argument(
        "--use_wandb",
        action=argparse.BooleanOptionalAction,
        default=training_config.use_wandb,
    )
    
    return parser.parse_args()

def train(args):

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
    step: int,
    wblogger: WandbLogger
):
    eval_data = np.memmap(args.eval_data_path, dtype=np.uint16, mode='r')

    batch = data.data_loading(eval_data, args.batch_size, args.context_length, args.device)
    sample = batch[0]
    target = batch[1]

    with torch.no_grad():
        logits = model(sample)

        ppl = loss.perplexity(logits,target)
        celoss = loss.cross_entropy(logits, target)

        wblogger.eval_log(celoss, ppl, step)
    return

def loop (
    args: argparse.Namespace,
    model: transformer.TransformerLM,
    optim: optimizer.AdamW,
):  
    wblogger = WandbLogger(args)
    # Semantics: finished training steps number
    global_step = 0

    # Lazy-load data
    training_data = np.memmap(args.training_data_path, dtype=np.uint16, mode='r')

    # Progress bar
    pbar = tqdm(total=args.total_step, desc="Training", unit="step")

    # Traing loop
    while global_step < args.total_step:
        batch = data.data_loading(training_data, args.batch_size, args.context_length, args.device)
        sample = batch[0]
        target = batch[1]

        logits = model(sample)

        celoss = loss.cross_entropy(logits, target)
        celoss.backward()

        optim.step()
        optim.zero_grad()

        global_step += 1
        pbar.update(1)
        wblogger.train_log(loss=celoss, lr=optim.param_groups[0]['lr'], step=global_step)

        if global_step % args.eval_every == 0:
            eval_model(args, model, global_step, wblogger)

        if global_step % args.save_every == 0:
            None

    checkpoint.save_checkpoint(model, optim, global_step, args.save_dir)
    pbar.close()
    wblogger.finish()

if __name__ == "__main__":
    logger.info("TRAINING START")
    args = parse_args()
    train(args)
