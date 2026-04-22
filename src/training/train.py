from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from config_manager import DataConfig, TrainingConfig
from model import DecoderOnlyTransformerLM
from run_artifacts import append_jsonl_record, save_checkpoint

from .data import build_dataloaders


@dataclass(slots=True)
class TrainingResult:
    """Summary of a completed training run.

    Extended description.

    Attributes:
        global_step: Number of completed optimization steps.
        train_loss: Training loss from the final optimization step.
        validation_loss: Validation loss from the most recent evaluation run.
        best_validation_loss: Best validation loss observed during training.
        artifact_dir: Run-artifact directory used for saving outputs, if any.
    """

    global_step: int
    train_loss: float
    validation_loss: float | None
    best_validation_loss: float | None = None
    artifact_dir: str | None = None


def build_checkpoint_payload(
    model: DecoderOnlyTransformerLM,
    global_step: int,
    train_loss: float,
    validation_loss: float | None,
) -> dict[str, object]:
    """Create the saved checkpoint payload for a training step.

    Args:
        model: Trained language model.
        global_step: Completed optimizer step count.
        train_loss: Training loss for the current step.
        validation_loss: Validation loss associated with the checkpoint.

    Returns:
        A serializable checkpoint payload dictionary.
    """
    return {
        "global_step": global_step,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
    }


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
    artifact_dir: str | Path | None = None,
) -> TrainingResult:
    """Train a decoder-only language model with AdamW and warmup.

    Args:
        model: Language model to optimize.
        text_tokenizer: Tokenizer instance used for text encoding.
        data_config: Data-path configuration.
        training_config: Training-loop configuration.
        artifact_dir: Optional run-artifact directory for saving training outputs.

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
    run_artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
    metrics_path = None
    latest_checkpoint_path = None
    best_checkpoint_path = None
    if run_artifact_dir is not None:
        metrics_path = run_artifact_dir / "metrics.jsonl"
        checkpoints_dir = run_artifact_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        latest_checkpoint_path = checkpoints_dir / "latest.pt"
        best_checkpoint_path = checkpoints_dir / "best.pt"

    train_iterator = iter(train_loader)
    last_train_loss = float("nan")
    last_valid_loss: float | None = None
    best_valid_loss: float | None = None

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
            if metrics_path is not None:
                append_jsonl_record(
                    metrics_path,
                    {
                        "lr": step_lr,
                        "step": step,
                        "train_loss": last_train_loss,
                        "type": "train",
                    },
                )

        if step % training_config.eval_interval == 0:
            last_valid_loss = evaluate(
                model=model,
                valid_loader=valid_loader,
                device=device,
                eval_steps=training_config.eval_steps,
            )
            print(f"step={step} valid_loss={last_valid_loss:.4f}")
            if metrics_path is not None:
                append_jsonl_record(
                    metrics_path,
                    {
                        "step": step,
                        "type": "eval",
                        "valid_loss": last_valid_loss,
                    },
                )
            if latest_checkpoint_path is not None:
                checkpoint_payload = build_checkpoint_payload(
                    model=model,
                    global_step=step,
                    train_loss=last_train_loss,
                    validation_loss=last_valid_loss,
                )
                save_checkpoint(latest_checkpoint_path, checkpoint_payload)
                if best_valid_loss is None or last_valid_loss < best_valid_loss:
                    best_valid_loss = last_valid_loss
                    if best_checkpoint_path is not None:
                        save_checkpoint(best_checkpoint_path, checkpoint_payload)
            elif best_valid_loss is None or last_valid_loss < best_valid_loss:
                best_valid_loss = last_valid_loss

    return TrainingResult(
        global_step=training_config.max_steps,
        train_loss=last_train_loss,
        validation_loss=last_valid_loss,
        best_validation_loss=best_valid_loss,
        artifact_dir=str(run_artifact_dir) if run_artifact_dir is not None else None,
    )
