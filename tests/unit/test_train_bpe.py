import pytest
from io import BytesIO
from tokenizer import train_bpe


def _toy_vocab(*tokens: bytes) -> train_bpe.Vocabulary:
    return {idx: token for idx, token in enumerate(tokens)}


def _byte_tokens(value: str | bytes) -> tuple[bytes, ...]:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return tuple(bytes([byte]) for byte in raw)


@pytest.fixture
def example_special_tokens()-> list[str]:
    return ["<PAD>", "<UNK>", "<EOS>", "<|endoftext|>"]

@pytest.fixture
def example_str()-> str:
    return "<BOS>Hello, world! <PAD>This is a test string<UNK> with special tokens.<EOS> <|endoftext|>"

@pytest.mark.unit
def test_uft8_vocab():
    vocab = train_bpe.uft8_vocab()
    assert len(vocab) == 256
    assert set(vocab.keys()) == set(range(256))
    for i in range(256):
        assert vocab[i] == bytes([i])
        assert len(vocab[i]) == 1

@pytest.mark.unit
def test_remove_special_tokens(example_str, example_special_tokens):
    results = train_bpe.remove_special_tokens(example_str, example_special_tokens)
    assert results[0] == "<BOS>Hello, world! "
    assert results[1] == "This is a test string"
    assert results[2] == " with special tokens."
    assert results[3] == " "
    assert results[4] == ""
    assert len(results) == 5


@pytest.mark.unit
def test_remove_special_tokens_with_empty_special_tokens_returns_original_text():
    text = "Hello world"
    assert train_bpe.remove_special_tokens(text, []) == [text]


@pytest.mark.unit
def test_pretokenize(example_str, example_special_tokens):
    pretokenized = train_bpe.pretokenize(example_str, example_special_tokens)

    assert pretokenized == [
        "<",
        "BOS",
        ">",
        "Hello",
        ",",
        " world",
        "!",
        " ",
        "This",
        " is",
        " a",
        " test",
        " string",
        " with",
        " special",
        " tokens",
        ".",
        " ",
    ]


@pytest.mark.unit
def test_pretokenization_frequency_table(example_str, example_special_tokens):
    fre_table = train_bpe.pretokenization_frequency_table(
        train_bpe.pretokenize(example_str, example_special_tokens)
    )
    assert fre_table[_byte_tokens(b"<")] == 1
    assert fre_table[_byte_tokens(b"BOS")] == 1
    assert fre_table[_byte_tokens(b">")] == 1
    assert fre_table[_byte_tokens(b"Hello")] == 1
    assert fre_table[_byte_tokens(b",")] == 1
    assert fre_table[_byte_tokens(b" world")] == 1
    assert fre_table[_byte_tokens(b"!")] == 1
    assert fre_table[_byte_tokens(b" ")] == 2
    assert fre_table[_byte_tokens(b"This")] == 1
    assert fre_table[_byte_tokens(b" is")] == 1
    assert fre_table[_byte_tokens(b" a")] == 1
    assert fre_table[_byte_tokens(b" test")] == 1
    assert fre_table[_byte_tokens(b" string")] == 1
    assert fre_table[_byte_tokens(b" with")] == 1
    assert fre_table[_byte_tokens(b" special")] == 1
    assert fre_table[_byte_tokens(b" tokens")] == 1
    assert fre_table[_byte_tokens(b".")] == 1


@pytest.mark.unit
def test_pretokenize_with_empty_special_tokens_counts_repeated_tokens():
    assert train_bpe.pretokenize("hi hi", []) == ["hi", " hi"]


@pytest.mark.unit
def test_pretokenization_frequency_table_aggregates_repeated_tokens():
    fre_table = train_bpe.pretokenization_frequency_table(
        ["Hello", " Hello", " Hello"]
    )
    assert fre_table[_byte_tokens(b"Hello")] == 1
    assert fre_table[_byte_tokens(b" Hello")] == 2


@pytest.mark.unit
def test_pretokenization_frequency_table_handles_unicode_and_numbers():
    fre_table = train_bpe.pretokenization_frequency_table(
        train_bpe.pretokenize("你好 123 你好", [])
    )
    assert fre_table[_byte_tokens("你好")] == 1
    assert fre_table[_byte_tokens(b" 123")] == 1
    assert fre_table[_byte_tokens(" 你好")] == 1


@pytest.mark.unit
def test_pretokenize_returns_empty_list_for_only_special_tokens():
    pretokenized = train_bpe.pretokenize("<PAD><UNK><PAD>", ["<PAD>", "<UNK>"])
    assert pretokenized == []


@pytest.mark.unit
def test_pretokenization_frequency_table_returns_empty_table_for_empty_pretokenization():
    fre_table = train_bpe.pretokenization_frequency_table([])
    assert fre_table == {}


@pytest.mark.unit
def test_find_chunk_boundaries_returns_only_start_and_end_when_split_token_is_absent():
    data = BytesIO(b"plain text without boundary")

    boundaries = train_bpe.find_chunk_boundaries(data, desired_num_chunks=4, split_special_token=b"<|endoftext|>")

    assert boundaries == [0, len(b"plain text without boundary")]


@pytest.mark.unit
def test_find_chunk_boundaries_moves_boundary_to_next_split_token():
    content = b"aaaaaaaaaa<|endoftext|>bbbb"
    data = BytesIO(content)

    boundaries = train_bpe.find_chunk_boundaries(data, desired_num_chunks=3, split_special_token=b"<|endoftext|>")

    assert boundaries == [0, 10, len(content)]


@pytest.mark.unit
def test_count_pairs_in_word_counts_overlapping_pairs():
    pair_counts = train_bpe.count_pairs_in_word((b"a", b"a", b"a"))

    assert pair_counts[(b"a", b"a")] == 2
    assert len(pair_counts) == 1


@pytest.mark.unit
def test_merge_pair_merges_left_to_right_without_reusing_bytes():
    merged = train_bpe.merge_pair((b"a", b"a"), (b"a", b"a", b"a"))

    assert merged == (b"aa", b"a")


@pytest.mark.unit
@pytest.mark.parametrize("target_size", [2, 1])
def test_merge_returns_same_vocab_when_target_size_is_not_greater(target_size):
    vocab = _toy_vocab(b"a", b"b")
    fre_table = {
        (b"a", b"b"): 3,
    }

    new_vocab, merges = train_bpe.merge(vocab.copy(), fre_table, target_size)

    assert new_vocab == vocab
    assert merges == []


@pytest.mark.unit
def test_merge_returns_same_vocab_when_no_pairs_are_available():
    vocab = _toy_vocab(b"a", b"b")
    fre_table = {
        (b"a",): 5,
        (b"b",): 1,
    }

    new_vocab, merges = train_bpe.merge(vocab.copy(), fre_table, vocab_size=4)

    assert new_vocab == vocab
    assert merges == []


@pytest.mark.unit
def test_merge_selects_pair_by_weighted_frequency():
    vocab = _toy_vocab(b"a", b"b", b"c", b"d", b"x", b"y")
    fre_table = {
        (b"a", b"b"): 4,
        (b"c", b"d"): 1,
        (b"x", b"c", b"d"): 1,
        (b"c", b"d", b"y"): 1,
    }

    new_vocab, merges = train_bpe.merge(vocab.copy(), fre_table, vocab_size=len(vocab) + 1)

    assert merges == [(b"a", b"b")]
    assert len(new_vocab) == len(vocab) + 1
    assert b"ab" in new_vocab.values()


@pytest.mark.unit
def test_merge_breaks_ties_lexicographically():
    vocab = _toy_vocab(b"a", b"b", b"c")
    fre_table = {
        (b"a", b"c"): 2,
        (b"b", b"c"): 2,
    }

    new_vocab, merges = train_bpe.merge(vocab.copy(), fre_table, vocab_size=len(vocab) + 1)

    assert merges == [(b"b", b"c")]
    assert len(new_vocab) == len(vocab) + 1
    assert b"bc" in new_vocab.values()


@pytest.mark.unit
def test_merge_performs_multiple_rounds_in_order():
    vocab = _toy_vocab(b"a", b"b", b"c")
    fre_table = {
        (b"a", b"b"): 4,
        (b"a", b"b", b"c"): 2,
    }

    new_vocab, merges = train_bpe.merge(vocab.copy(), fre_table, vocab_size=len(vocab) + 2)

    assert merges == [
        (b"a", b"b"),
        (b"ab", b"c"),
    ]
    assert len(new_vocab) == len(vocab) + 2
    assert b"ab" in new_vocab.values()
    assert b"abc" in new_vocab.values()


@pytest.mark.unit
def test_merge_updates_counts_across_multiple_words():
    vocab = _toy_vocab(b"a", b"b", b"c")
    fre_table = {
        (b"a", b"b", b"c"): 2,
        (b"a", b"b"): 1,
    }

    new_vocab, merges = train_bpe.merge(vocab.copy(), fre_table, vocab_size=len(vocab) + 2)

    assert merges == [(b"a", b"b"), (b"ab", b"c")]
    assert len(new_vocab) == len(vocab) + 2
    assert b"ab" in new_vocab.values()
    assert b"abc" in new_vocab.values()


@pytest.mark.unit
def test_merge_handles_overlapping_pairs_left_to_right():
    vocab = _toy_vocab(b"a")
    fre_table = {
        (b"a", b"a", b"a", b"a"): 1,
    }

    new_vocab, merges = train_bpe.merge(vocab.copy(), fre_table, vocab_size=len(vocab) + 2)

    assert merges == [
        (b"a", b"a"),
        (b"aa", b"aa"),
    ]
    assert len(new_vocab) == len(vocab) + 2
    assert b"aa" in new_vocab.values()
    assert b"aaaa" in new_vocab.values()
