import pytest

from tokenizer.tokenizer import tokenizer


def _base_vocab() -> dict[int, bytes]:
    return {idx: bytes([idx]) for idx in range(256)}


def _vocab_with_extra_tokens(*tokens: bytes) -> dict[int, bytes]:
    vocab = _base_vocab()
    for index, token in enumerate(tokens, start=256):
        vocab[index] = token
    return vocab


def _utf8_token_ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "special_tokens"),
    [
        ("abc", []),
        (" hi", []),
        ("a\nb", []),
        ("你好", []),
        ("🙂", []),
        ("aa<PAD>aa", ["<PAD>"]),
    ],
)
def test_encode_without_merges_returns_utf8_byte_ids(text, special_tokens):
    tk = tokenizer(vocab=_base_vocab(), merges=[], special_tokens=special_tokens)

    assert tk.encode(text) == _utf8_token_ids(text)


@pytest.mark.unit
def test_encode_returns_empty_list_for_empty_text():
    tk = tokenizer(vocab=_base_vocab(), merges=[], special_tokens=[])

    assert tk.encode("") == []


@pytest.mark.unit
def test_encode_default_special_tokens_matches_empty_list():
    text = "abc"
    default_tk = tokenizer(vocab=_base_vocab(), merges=[])
    explicit_tk = tokenizer(vocab=_base_vocab(), merges=[], special_tokens=[])

    assert default_tk.encode(text) == explicit_tk.encode(text) == _utf8_token_ids(text)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("aaaa", [256, 256]),
        ("aaa", [256, 97]),
        ("a", [97]),
    ],
)
def test_encode_applies_single_merge_repeatedly_left_to_right(text, expected):
    tk = tokenizer(
        vocab=_vocab_with_extra_tokens(b"aa"),
        merges=[(b"a", b"a")],
        special_tokens=[],
    )

    assert tk.encode(text) == expected


@pytest.mark.unit
def test_encode_applies_chained_merges_when_merge_order_allows_it():
    tk = tokenizer(
        vocab=_vocab_with_extra_tokens(b"ab", b"abc"),
        merges=[(b"a", b"b"), (b"ab", b"c")],
        special_tokens=[],
    )

    assert tk.encode("abc") == [257]


@pytest.mark.unit
def test_encode_respects_merge_list_order_for_chained_merges():
    tk = tokenizer(
        vocab=_vocab_with_extra_tokens(b"ab", b"abc"),
        merges=[(b"ab", b"c"), (b"a", b"b")],
        special_tokens=[],
    )

    assert tk.encode("abc") == [256, 99]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "merge", "extra_token", "expected"),
    [
        ("a<PAD>", (b"a", b"<"), b"a<", [97, 60, 80, 65, 68, 62]),
        ("<PAD>a", (b">", b"a"), b">a", [60, 80, 65, 68, 62, 97]),
    ],
)
def test_encode_treats_special_tokens_as_hard_merge_boundaries(
    text, merge, extra_token, expected
):
    tk = tokenizer(
        vocab=_vocab_with_extra_tokens(extra_token),
        merges=[merge],
        special_tokens=["<PAD>"],
    )

    assert tk.encode(text) == expected


@pytest.mark.unit
def test_encode_can_merge_inside_special_token_text():
    tk = tokenizer(
        vocab=_vocab_with_extra_tokens(b"PA"),
        merges=[(b"P", b"A")],
        special_tokens=["<PAD>"],
    )

    assert tk.encode("a<PAD>a") == [97, 60, 256, 68, 62, 97]


@pytest.mark.unit
def test_encode_raises_key_error_when_merged_token_is_missing_from_vocab():
    tk = tokenizer(
        vocab=_base_vocab(),
        merges=[(b"a", b"a")],
        special_tokens=[],
    )

    with pytest.raises(KeyError, match=r"b'aa'"):
        tk.encode("aa")
