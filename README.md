# NanoLLM

NanoLLM is a small from-scratch language-model project. The repository currently includes:

- a byte-level BPE tokenizer in `src/tokenizer/`
- a decoder-only Transformer LM in `src/model/`
- a minimal training stack in `src/training/`

## Training

The training loop uses:

- shifted next-token cross-entropy
- `torch.optim.AdamW`
- linear warmup followed by a constant learning rate
- streaming text reads instead of loading the full corpus into memory

Training depends on tokenizer artifacts already existing at the paths configured under `bpe:` in `config/default.yaml`.

Run training with:

```bash
uv run python scripts/train_lm.py
```

## Configuration

The default repository config lives in `config/default.yaml` and contains:

- `bpe`: tokenizer artifact paths and special tokens
- `data`: corpus paths for OWT and TinyStories
- `model`: decoder-only Transformer hyper-parameters
- `training`: batch size, learning rate, warmup, evaluation cadence, and device

## Documentation

More detail is available in:

- `docs/tokenizer.md`
- `docs/decoder_lm.md`
- `docs/training.md`
