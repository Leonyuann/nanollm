"""Combine terminal, local, and Weights & Biases logging."""

import json
from pathlib import Path

from loguru import logger
import wandb
from datetime import datetime
from zoneinfo import ZoneInfo


def local_record(file: str, record: dict) -> None:
    """Append one training record to a local JSON array.

    Args:
        file: Path to the JSON log file.
        record: Training metadata to append.

    Raises:
        ValueError: If the existing file is not a JSON array.
    """
    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists() or path.stat().st_size == 0:
        records = []
    else:
        with path.open("r", encoding="utf-8") as log_file:
            records = json.load(log_file)

        if not isinstance(records, list):
            raise ValueError(f"Local log must contain a JSON array: {path}")

    records.append(record)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as log_file:
        json.dump(records, log_file, indent=4)
        log_file.write("\n")
    temporary_path.replace(path)


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

    def train_log(self, data, step):
        if not self.enabled: 
            return 
        
        self.run.log(
            data={f"Training/{key}": value for key, value in data.items()},
            step=step,
        )

    def eval_log(self, data:dict, step:int):
        if not self.enabled:
            return
        
        self.run.log(
            data= {f"Eval/{key}" : value for key, value in data.items()},
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