import pytest
import torch

from config_manager import DataConfig
from tokenizer.tokenizer import tokenizer
from training import (
    TokenSequenceIterableDataset,
    TrainingConfig,
    WarmupConstantScheduler,
    compute_next_token_loss,
)
from training.data import resolve_dataset_paths


def _base_vocab() -> dict[int, bytes]:
    """Create the byte-level base vocabulary used by tokenizer tests.

    Args:
        None.

    Returns:
        A mapping from token id to raw byte value.
    """
    return {idx: bytes([idx]) for idx in range(256)}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"dataset": "books"}, "dataset must be one of"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"max_steps": 0}, "max_steps must be positive"),
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"warmup_steps": -1}, "warmup_steps must be non-negative"),
        ({"eval_interval": 0}, "eval_interval must be positive"),
        ({"eval_steps": 0}, "eval_steps must be positive"),
        ({"device": "mps"}, "device must be one of"),
    ],
)
def test_training_config_rejects_invalid_values(overrides, expected_message):
    defaults = {
        "dataset": "tinystories",
        "batch_size": 2,
        "max_steps": 3,
        "learning_rate": 1e-3,
        "warmup_steps": 1,
        "log_interval": 1,
        "eval_interval": 1,
        "eval_steps": 1,
        "device": "cpu",
        "seed": 0,
    }
    defaults.update(overrides)

    with pytest.raises(ValueError, match=expected_message):
        TrainingConfig(**defaults)


@pytest.mark.unit
def test_resolve_dataset_paths_returns_expected_tinystories_paths():
    data_config = DataConfig(
        owt_train_path="owt-train.txt",
        owt_valid_path="owt-valid.txt",
        TinyStories_train_path="tiny-train.txt",
        TinyStories_valid_path="tiny-valid.txt",
    )

    train_path, valid_path = resolve_dataset_paths(data_config, "tinystories")

    assert train_path == "tiny-train.txt"
    assert valid_path == "tiny-valid.txt"


@pytest.mark.unit
def test_streaming_dataset_yields_fixed_windows_and_can_cross_lines(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("ab\ncd", encoding="utf-8")
    text_tokenizer = tokenizer(vocab=_base_vocab(), merges=[], special_tokens=[])

    dataset = TokenSequenceIterableDataset(
        file_path=str(corpus_path),
        text_tokenizer=text_tokenizer,
        seq_len=3,
        repeat=False,
    )

    windows = [window.tolist() for window in dataset]

    assert windows == [[ord("a"), ord("b"), ord("\n"), ord("c")]]


@pytest.mark.unit
def test_streaming_dataset_repeats_training_data_and_drops_short_residuals(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("abcdefg", encoding="utf-8")
    text_tokenizer = tokenizer(vocab=_base_vocab(), merges=[], special_tokens=[])

    dataset = TokenSequenceIterableDataset(
        file_path=str(corpus_path),
        text_tokenizer=text_tokenizer,
        seq_len=2,
        repeat=True,
    )
    iterator = iter(dataset)
    windows = [next(iterator).tolist() for _ in range(3)]

    assert windows == [
        [ord("a"), ord("b"), ord("c")],
        [ord("d"), ord("e"), ord("f")],
        [ord("a"), ord("b"), ord("c")],
    ]


@pytest.mark.unit
def test_streaming_dataset_validation_iteration_stops_at_end_of_file(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("abcdef", encoding="utf-8")
    text_tokenizer = tokenizer(vocab=_base_vocab(), merges=[], special_tokens=[])

    dataset = TokenSequenceIterableDataset(
        file_path=str(corpus_path),
        text_tokenizer=text_tokenizer,
        seq_len=2,
        repeat=False,
    )
    iterator = iter(dataset)

    assert next(iterator).tolist() == [ord("a"), ord("b"), ord("c")]
    assert next(iterator).tolist() == [ord("d"), ord("e"), ord("f")]
    with pytest.raises(StopIteration):
        next(iterator)


@pytest.mark.unit
def test_compute_next_token_loss_matches_manual_cross_entropy():
    logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0]]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0, 1]], dtype=torch.long)

    log_probs = torch.log_softmax(logits, dim=-1)
    expected = -0.5 * (log_probs[0, 0, 0] + log_probs[0, 1, 1])
    loss = compute_next_token_loss(logits, targets)

    assert torch.allclose(loss, expected)


@pytest.mark.unit
def test_warmup_scheduler_increases_then_stays_constant():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = WarmupConstantScheduler(optimizer=optimizer, warmup_steps=4)

    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(5):
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert learning_rates[4:] == pytest.approx([1.0, 1.0])


@pytest.mark.unit
def test_warmup_scheduler_is_constant_when_warmup_is_disabled():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = WarmupConstantScheduler(optimizer=optimizer, warmup_steps=0)

    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(3):
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates == pytest.approx([1.0, 1.0, 1.0, 1.0])
