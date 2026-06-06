# Tokenizer

## Overview

This project implements a compact byte-pair encoding (BPE) tokenizer in `src/tokenizer/`. The codebase covers the full lifecycle of a small tokenizer system:

- training a BPE vocabulary from raw text
- pretokenizing text with special-token-aware boundaries
- encoding strings into token IDs
- decoding token IDs back into UTF-8 text
- streaming tokenization over iterables

The implementation is intentionally simple and explicit. It favors readability and testability over aggressive optimization, which makes it a good reference implementation for learning how a byte-level BPE tokenizer works.

## Module Map

### `src/tokenizer/train_bpe.py`

Contains the training pipeline:

- file chunking for parallel counting
- special-token-aware pretokenization
- frequency table construction
- iterative pair merging
- end-to-end `train_bpe(...)`

### `src/tokenizer/tokenizer.py`

Contains the runtime tokenizer class:

- `tokenizer.from_files(...)`
- `tokenizer.encode(...)`
- `tokenizer.encode_iterable(...)`
- `tokenizer.decode(...)`

### `src/tokenizer/types.py`

Defines the shared type aliases:

- `Vocabulary = dict[int, bytes]`
- `Merges = list[tuple[bytes, bytes]]`

## Design Principles

The tokenizer follows a few clear implementation choices:

1. It is byte-level at the base vocabulary. The initial vocabulary always contains the 256 single-byte values.
2. It learns merges on top of bytes, not Unicode code points.
3. Special tokens act as hard boundaries during pretokenization, so merges do not cross those boundaries.
4. Merge order matters. Encoding applies merges in the exact order learned during training.
5. Decoding is loss-tolerant. Invalid UTF-8 sequences are decoded with `errors="replace"`.

This is close to the mental model used by modern BPE tokenizers, while staying small enough to inspect end to end.

## Training Pipeline

The training entry point is
`train_bpe(input_path, vocab_size, special_tokens, show_progress=False)`.
The training script reports only BPE merge progress with `tqdm`.

### Step 1: Initialize the vocabulary

Training starts from `uft8_vocab()`, which creates the base vocabulary:

- token IDs `0..255`
- one byte per token

After that, each configured special token is appended to the vocabulary as a full UTF-8 byte string.

This means special tokens exist in the learned vocabulary. However, the current runtime encoder does not directly map a matched special-token substring to that single vocabulary ID.

### Step 2: Split the corpus into safe chunks

`find_chunk_boundaries(...)` divides the input file into roughly even byte ranges for multiprocessing. The boundaries are adjusted so they land on the next occurrence of a split token, currently `b"<|endoftext|>"`.

This avoids counting across document boundaries when the corpus contains that delimiter. Workers receive byte ranges and read their chunks directly from the corpus, so the parent process does not retain or serialize the full decoded chunks.

### Step 3: Pretokenize each chunk

`pretokenize(...)` first removes configured special tokens from the chunk by treating them as hard separators. It then applies a regex-based pretokenization pattern that keeps:

- words
- numbers
- punctuation
- whitespace spans

Leading spaces are preserved inside many tokens, which matches the behavior expected from GPT-style tokenizers.

### Step 4: Build a frequency table

`pretokenization_frequency_table(...)` converts every pretokenized string into a tuple of UTF-8 bytes and counts how often that tuple appears.

At this stage, the system is still purely byte-level.

### Step 5: Repeatedly merge the most frequent pair

`merge(...)` performs standard BPE growth:

- count adjacent byte-pair occurrences
- choose the most frequent pair
- break ties lexicographically
- add the merged byte sequence to the vocabulary
- update only the affected words and pair counts

This continues until the requested vocabulary size is reached or no mergeable pairs remain.

## Runtime Tokenization

The runtime class is `tokenizer`.

### `encode(text)`

Encoding works as follows:

1. Pretokenize the input text.
2. Convert each pretoken into a tuple of single-byte tokens.
3. Replay the learned merge list from first to last.
4. Map the final byte tokens into vocabulary IDs.

Important behavior:

- with no learned merges, encoding falls back to raw UTF-8 byte IDs
- merges are applied left to right
- chained merges only happen when earlier merges create the input required by later ones
- special-token boundaries block merges from crossing through them
- special tokens are currently treated as boundaries during pretokenization, not as guaranteed single-token outputs

### `encode_iterable(iterable)`

`encode_iterable(...)` is a lazy generator. It does not materialize the full input or output sequence up front. This is useful for streaming tokenization over large datasets.

### `decode(ids)`

Decoding concatenates the byte sequence for each token ID and then decodes the full byte string as UTF-8.

## Configuration

Tokenizer-related configuration is loaded from `config/default.yaml` through `src/config_manager.py`.

The relevant BPE fields are:

```yaml
bpe:
  num_process: 4
  vocab_size: 10000
  special_tokens: ["<|endoftext|>"]
  vocab_path: "outputs/artifacts/vocab.txt"
  merge_path: "outputs/artifacts/merges.txt"
```

The current implementation uses configuration in two places:

- training uses `num_process`
- runtime tokenizer instances use the `special_tokens` passed to the constructor or `from_files(...)`

That second point is important: special tokens are runtime behavior, so the tokenizer you load should receive the same special token list that was used when the vocabulary was trained or assembled.

## Artifact Format

`tokenizer.from_files(...)` expects serialized artifacts in a machine-readable format:

### Vocabulary file

One token per line:

```text
token_id<TAB>hex_bytes
```

Example:

```text
97	61
98	62
256	6162
```

### Merge file

One merge pair per line:

```text
left_hex_bytes<TAB>right_hex_bytes
```

Example:

```text
61	62
```

This format is different from the current `main.py` output, which writes human-readable text such as `97 -> a`. Those files are convenient for inspection, but they are not directly compatible with `tokenizer.from_files(...)` as written today.

## Usage Examples

### Train a tokenizer

```python
from tokenizer import train_bpe

vocab, merges = train_bpe.train_bpe(
    input_path="data/owt_train.txt",
    vocab_size=1000,
    special_tokens=["<|endoftext|>"],
)
```

### Build a runtime tokenizer directly

```python
from tokenizer.tokenizer import tokenizer

tk = tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])
ids = tk.encode("Hello world")
text = tk.decode(ids)
```

### Stream tokenization over multiple inputs

```python
token_stream = tk.encode_iterable(["Hello", " world", "\n"])
ids = list(token_stream)
```

## Behavior Verified by Tests

The tests under `tests/unit/` and `tests/integration/` document the intended behavior well. They verify, among other things:

- UTF-8 byte fallback when no merges exist
- repeated left-to-right merge application
- merge-order sensitivity
- special-token boundary handling
- lazy iteration semantics for `encode_iterable(...)`
- file-based loading with `from_files(...)`
- weighted pair selection and lexicographic tie-breaking during training
- end-to-end vocabulary growth from a corpus file

If you need to understand expected edge-case behavior, the tests are the most precise companion to this document.

## Summary

`src/tokenizer/` provides a compact, test-backed byte-level BPE tokenizer implementation with:

- a readable training pipeline
- a practical runtime API
- multiprocessing-aware corpus counting
- explicit special-token handling
- incremental evolution reflected clearly in git history

For contributors, the fastest path to understanding the system is:

1. read `train_bpe.py`
2. read `tokenizer.py`
3. inspect the tokenizer tests
4. keep the config and artifact-format caveats in mind
