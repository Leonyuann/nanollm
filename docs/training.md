# Training

## Overview

This project now includes a minimal language-model training stack in `src/training/`.

The implementation is intentionally small and explicit:

- streaming token windows from text files
- next-token cross-entropy on every position
- `torch.optim.AdamW`
- linear warmup followed by a constant learning rate
- periodic validation on the configured validation split

The training code does not modify tokenizer behavior. It only consumes the current runtime tokenizer through `tokenizer.from_files(...)` and `encode_iterable(...)`.

## Module Map

### `src/training/config.py`

Defines `TrainingConfig`, which validates the training loop configuration.

### `src/training/data.py`

Contains the streaming dataset and dataloader helpers:

- `TokenSequenceIterableDataset`
- `resolve_dataset_paths(...)`
- `build_dataloaders(...)`

### `src/training/train.py`

Contains the training logic:

- `WarmupConstantScheduler`
- `compute_next_token_loss(...)`
- `evaluate(...)`
- `run_training(...)`

## Configuration

Training configuration is loaded from `config/default.yaml` through `config_manager.load_TrainingConfig()`.

```yaml
training:
  dataset: "tinystories"
  batch_size: 8
  max_steps: 1000
  learning_rate: 3.0e-4
  warmup_steps: 100
  log_interval: 10
  eval_interval: 100
  eval_steps: 20
  device: "auto"
  seed: 42
```

Field meanings:

- `dataset`: one of `tinystories` or `owt`
- `batch_size`: number of token windows per training step
- `max_steps`: number of optimizer steps to run
- `learning_rate`: AdamW base learning rate
- `warmup_steps`: number of linear warmup steps
- `log_interval`: interval for printing training loss and learning rate
- `eval_interval`: interval for running validation
- `eval_steps`: maximum number of validation batches per evaluation
- `device`: one of `auto`, `cpu`, or `cuda`
- `seed`: PyTorch RNG seed

The sequence length is inherited from `model.max_seq_len`.

## Data Flow

Training and validation both read plain-text corpora from the paths selected by `training.dataset`.

The data pipeline works as follows:

1. read the corpus lazily line by line
2. encode the text stream with `tokenizer.encode_iterable(...)`
3. accumulate a buffer of token ids
4. emit fixed windows of length `max_seq_len + 1`
5. split each batch into:
   - `input_ids = batch[:, :-1]`
   - `targets = batch[:, 1:]`

Important behavior:

- training repeats forever by reopening the training file after EOF
- validation stops at EOF
- windows are contiguous and non-overlapping within a pass
- no padding is used
- leftover tokens shorter than a full window are dropped

## Loss And Optimization

The training loop uses standard next-token cross-entropy:

```python
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
)
```

Optimization uses:

```python
torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
```

The learning-rate schedule is:

- constant from the start when `warmup_steps == 0`
- otherwise linearly increase from step 1 to `warmup_steps`
- stay constant at the base learning rate after warmup

## Running Training

Before training, make sure the tokenizer artifact files referenced by `bpe.vocab_path` and `bpe.merge_path` already exist.

Run:

```bash
uv run python scripts/train_lm.py
```

Optionally:

```bash
uv run python scripts/train_lm.py --config config/default.yaml
```
