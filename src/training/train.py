from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from config_manager import DataConfig, TrainingConfig
from model import DecoderOnlyTransformerLM

from .data import build_dataloaders


@dataclass(slots=True)
class TrainingResult:
    """Summary of a completed training run.

    Extended description.

    Attributes:
        global_step: Number of completed optimization steps.
        train_loss: Training loss from the final optimization step.
        validation_loss: Validation loss from the most recent evaluation run.
    """

    global_step: int
    train_loss: float
    validation_loss: float | None


class WarmupConstantScheduler:
    """Linear warmup scheduler followed by a constant learning rate.

    Extended description.

    Attributes:
        optimizer: Optimizer whose learning rate is updated in place.
        warmup_steps: Number of warmup optimization steps.
        base_lrs: Target learning rates for each optimizer parameter group.
        step_count: Number of completed optimizer steps.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int) -> None:
        """Initialize the scheduler and set the first-step learning rate.

        Args:
            optimizer: Optimizer to control.
            warmup_steps: Number of warmup optimization steps.

        Returns:
            None.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0
        self._set_lrs(self._lr_scale_for_step(1))

    def _lr_scale_for_step(self, step_number: int) -> float:
        """Compute the multiplicative LR scale for an optimization step.

        Args:
            step_number: One-indexed optimization step number.

        Returns:
            A scaling factor relative to the optimizer base learning rate.
        """
        if self.warmup_steps == 0:
            return 1.0
        if step_number <= self.warmup_steps:
            return step_number / self.warmup_steps
        return 1.0

    def _set_lrs(self, scale: float) -> None:
        """Update all optimizer parameter groups with a shared LR scale.

        Args:
            scale: Multiplicative factor applied to each base learning rate.

        Returns:
            None.
        """
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            param_group["lr"] = base_lr * scale

    def step(self) -> None:
        """Advance the scheduler to the learning rate for the next step.

        Args:
            None.

        Returns:
            None.
        """
        self.step_count += 1
        self._set_lrs(self._lr_scale_for_step(self.step_count + 1))

    def get_last_lr(self) -> list[float]:
        """Return the currently active learning rates.

        Args:
            None.

        Returns:
            The active learning rate for each optimizer parameter group.
        """
        return [group["lr"] for group in self.optimizer.param_groups]


def compute_next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute mean next-token cross-entropy over all batch positions.

    Args:
        logits: Model logits with shape ``[batch, seq_len, vocab_size]``.
        targets: Shifted token targets with shape ``[batch, seq_len]``.

    Returns:
        A scalar loss tensor.
    """
    vocab_size = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))


def resolve_device(device_name: str) -> torch.device:
    """Resolve a configured device alias into a concrete PyTorch device.

    Args:
        device_name: Requested device alias.

    Returns:
        The resolved torch device.
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_random_seed(seed: int) -> None:
    """Seed PyTorch for deterministic training-loop setup.

    Args:
        seed: Random seed used for PyTorch RNGs.

    Returns:
        None.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: DecoderOnlyTransformerLM,
    valid_loader: torch.utils.data.DataLoader,
    device: torch.device,
    eval_steps: int,
) -> float:
    """Run a bounded validation pass and return the mean loss.

    Args:
        model: Language model under evaluation.
        valid_loader: Validation dataloader.
        device: Device on which evaluation should run.
        eval_steps: Maximum number of validation batches to evaluate.

    Returns:
        The mean validation loss across the evaluated batches.

    Raises:
        ValueError: If the validation loader does not yield any batches.
    """
    model.eval()
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(valid_loader, start=1):
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(input_ids)
            loss = compute_next_token_loss(logits, targets)
            total_loss += loss.item()
            batch_count += 1

            if batch_index >= eval_steps:
                break

    if batch_count == 0:
        raise ValueError("validation loader did not yield any batches")

    model.train()
    return total_loss / batch_count


def run_training(
    model: DecoderOnlyTransformerLM,
    text_tokenizer,
    data_config: DataConfig,
    training_config: TrainingConfig,
) -> TrainingResult:
    """Train a decoder-only language model with AdamW and warmup.

    Args:
        model: Language model to optimize.
        text_tokenizer: Tokenizer instance used for text encoding.
        data_config: Data-path configuration.
        training_config: Training-loop configuration.

    Returns:
        A summary of the completed training run.
    """
    set_random_seed(training_config.seed)

    device = resolve_device(training_config.device)
    model = model.to(device)
    train_loader, valid_loader = build_dataloaders(
        data_config=data_config,
        training_config=training_config,
        text_tokenizer=text_tokenizer,
        seq_len=model.config.max_seq_len,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
    )
    scheduler = WarmupConstantScheduler(
        optimizer=optimizer,
        warmup_steps=training_config.warmup_steps,
    )

    train_iterator = iter(train_loader)
    last_train_loss = float("nan")
    last_valid_loss: float | None = None

    model.train()
    for step in range(1, training_config.max_steps + 1):
        step_lr = optimizer.param_groups[0]["lr"]
        batch = next(train_iterator).to(device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(input_ids)
        loss = compute_next_token_loss(logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        last_train_loss = loss.item()

        if step % training_config.log_interval == 0:
            print(f"step={step} train_loss={last_train_loss:.4f} lr={step_lr:.6g}")

        if step % training_config.eval_interval == 0:
            last_valid_loss = evaluate(
                model=model,
                valid_loader=valid_loader,
                device=device,
                eval_steps=training_config.eval_steps,
            )
            print(f"step={step} valid_loss={last_valid_loss:.4f}")

    return TrainingResult(
        global_step=training_config.max_steps,
        train_loss=last_train_loss,
        validation_loss=last_valid_loss,
    )
