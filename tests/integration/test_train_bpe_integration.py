import pytest

from tokenizer import train_bpe


@pytest.mark.integration
def test_train_bpe_learns_single_merge_from_file(tmp_path):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("aa aa", encoding="utf-8")

    vocab, merges = train_bpe.train_bpe(
        input_path=str(input_path),
        vocab_size=257,
        special_tokens=[],
    )

    assert len(vocab) == 257
    assert merges == [(b"a", b"a")]
    assert vocab[256] == b"aa"


@pytest.mark.integration
def test_train_bpe_keeps_special_tokens_as_boundaries_and_in_vocab(tmp_path):
    input_path = tmp_path / "corpus_with_special.txt"
    input_path.write_text("aa<PAD>aa", encoding="utf-8")

    vocab, merges = train_bpe.train_bpe(
        input_path=str(input_path),
        vocab_size=258,
        special_tokens=["<PAD>"],
    )

    assert len(vocab) == 258
    assert vocab[256] == b"<PAD>"
    assert vocab[257] == b"aa"
    assert merges == [(b"a", b"a")]


@pytest.mark.integration
def test_train_bpe_stops_when_requested_vocab_size_is_already_reached(tmp_path):
    input_path = tmp_path / "no_growth.txt"
    input_path.write_text("aa aa", encoding="utf-8")

    vocab, merges = train_bpe.train_bpe(
        input_path=str(input_path),
        vocab_size=256,
        special_tokens=[],
    )

    assert len(vocab) == 256
    assert merges == []
    assert b"aa" not in vocab.values()
