# NanoLLM

NanoLLM is a small from-scratch language-model project. The repository currently includes:

- a byte-level BPE tokenizer in `src/tokenizer/`
- a decoder-only Transformer LM in `src/model/`
- a minimal training stack in `src/training/`
- workflow scripts for saved runs, standalone evaluation, and greedy generation in `scripts/`

The project stays intentionally small: model and tokenizer code remain explicit, while workflow concerns are handled by simple scripts.

## Workflow

The minimal end-to-end workflow is:

1. download data
2. train tokenizer artifacts
3. train the LM and save a run directory
4. evaluate the saved checkpoint on `owt_valid.txt`
5. generate a 256-token sample
6. optionally package a code-only submission zip

Download the datasets with:

```bash
bash scripts/download_data.sh
```

Run the full pipeline with:

```bash
uv run python scripts/run_pipeline.py --run-name demo-run
```

Or run each step explicitly:

```bash
uv run python scripts/train_tokenizer.py --input data/owt_train.txt
uv run python scripts/train_lm.py --run-name demo-run
uv run python scripts/evaluate_lm.py --run-dir outputs/runs/demo-run
uv run python scripts/generate_text.py --run-dir outputs/runs/demo-run
uv run python scripts/package_submission.py --output code.zip
```

## Training

The training loop uses:

- shifted next-token cross-entropy
- `torch.optim.AdamW`
- linear warmup followed by a constant learning rate
- streaming text reads instead of loading the full corpus into memory
- optional run artifact saving without changing the training numerics

Training depends on tokenizer artifacts already existing at the paths configured under `bpe:` in `config/default.yaml`.

Train tokenizer artifacts with:

```bash
uv run python scripts/train_tokenizer.py --input data/owt_train.txt
```

Run training with:

```bash
uv run python scripts/train_lm.py --run-name demo-run
```

## Configuration

The default repository config lives in `config/default.yaml` and contains:

- `bpe`: tokenizer artifact paths and special tokens
- `data`: corpus paths for OWT and TinyStories
- `artifacts`: run artifact root directory
- `model`: decoder-only Transformer hyper-parameters
- `training`: batch size, learning rate, warmup, evaluation cadence, and device

## Run Artifacts

Each training run is saved under `outputs/runs/<run_name>/` and contains:

- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `metrics.jsonl`
- `config.snapshot.yaml`
- `tokenizer/vocab.txt`
- `tokenizer/merges.txt`
- `eval/eval_owt.json`
- `samples/sample_256.txt`
- `summary.json`

These files are the main sources for report-ready metrics and reproducible reruns.

## Documentation

More detail is available in:

- `docs/tokenizer.md`
- `docs/decoder_lm.md`
- `docs/training.md`
