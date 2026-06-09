""" Combine logger and Wanbd into one logger
"""
from loguru import logger
import wandb
from datetime import datetime
from zoneinfo import ZoneInfo

class WandbLogger:
    def __init__(self, args):
        self.cfg = args
        self.enabled = args.use_wandb

        # Check if enable wandb
        if self.enabled:
            self.run = wandb.init(
                project="LLM-from-scratch",
                name=self.__build_run_name(),
                config=args,
            )

        # Logs in terminal
        logger.info(f"Starting training with the arguments")
        for k, v in vars(self.cfg).items():
            logger.info(f"{k:30} {v}")
        logger.info("*" * 40)

    def train_log(self, loss, lr, step):
        if not self.enabled: 
            return 
        
        self.run.log(
            data={
                "Training/loss": loss,
                "Training/lr": lr,
            },
            step=step,
        )

    def eval_log(self, loss, ppl, step):
        if not self.enabled:
            return
        
        self.run.log(
            data={
                "Eval/loss": loss,
                "Eval/ppl": ppl
            },
            step=step
        )

    def finish(self):
        if self.enabled:
            self.run.finish()
        
        logger.info("*" * 40)
        logger.info("Training finish.")
        logger.info("*" * 40)


    def __build_run_name(self):
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%m%d-%H%M")

        return (
            f"D{self.cfg.d_model}-"
            f"L{self.cfg.num_layers}-"
            f"F{self.cfg.d_ff}-"
            f"LR{self.cfg.lr}-"
            f"BS{self.cfg.batch_size}-"
            f"{timestamp}-"
        )