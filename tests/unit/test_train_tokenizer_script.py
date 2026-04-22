from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

import config_manager
from tokenizer.tokenizer import tokenizer


def _load_train_tokenizer_script() -> ModuleType:
    """Load the tokenizer training script as an importable module for tests.

    Args:
        None.

    Returns:
        The loaded script module.
    """
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "train_tokenizer.py"
    spec = importlib.util.spec_from_file_location("train_tokenizer_script", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_write_vocab_and_merge_files_use_machine_readable_hex_format(tmp_path):
    script_module = _load_train_tokenizer_script()
    vocab_path = tmp_path / "artifacts" / "vocab.txt"
    merge_path = tmp_path / "artifacts" / "merges.txt"

    script_module.write_vocab_file(
        vocab_path,
        {
            1: b"b",
            0: b"a",
            2: "你好".encode("utf-8"),
        },
    )
    script_module.write_merge_file(
        merge_path,
        [
            (b"a", b"b"),
            ("你".encode("utf-8"), "好".encode("utf-8")),
        ],
    )

    assert vocab_path.read_text(encoding="utf-8") == "0\t61\n1\t62\n2\te4bda0e5a5bd\n"
    assert merge_path.read_text(encoding="utf-8") == "61\t62\ne4bda0\te5a5bd\n"


@pytest.mark.integration
def test_train_tokenizer_script_main_trains_and_writes_loadable_artifacts(tmp_path, monkeypatch):
    script_module = _load_train_tokenizer_script()
    input_path = tmp_path / "corpus.txt"
    config_path = tmp_path / "config.yaml"
    vocab_path = tmp_path / "outputs" / "vocab.txt"
    merge_path = tmp_path / "outputs" / "merges.txt"

    input_path.write_text("aa aa", encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                "bpe:",
                "  num_process: 1",
                "  vocab_size: 257",
                "  special_tokens: []",
                f'  vocab_path: "{vocab_path}"',
                f'  merge_path: "{merge_path}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_manager, "config_path", "config/default.yaml")

    exit_code = script_module.main(
        [
            "--config",
            str(config_path),
            "--input",
            str(input_path),
        ]
    )

    assert exit_code == 0
    assert vocab_path.exists()
    assert merge_path.exists()

    trained_tokenizer = tokenizer.from_files(
        str(vocab_path),
        str(merge_path),
        special_tokens=[],
    )

    assert trained_tokenizer.merges == [(b"a", b"a")]
    assert trained_tokenizer.vocab[256] == b"aa"
    assert trained_tokenizer.encode("aa") == [256]
