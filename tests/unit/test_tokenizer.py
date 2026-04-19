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


@pytest.mark.unit
def test_encode_iterable_flattens_encoded_strings_in_order():
    tk = tokenizer(
        vocab=_vocab_with_extra_tokens(b"aa"),
        merges=[(b"a", b"a")],
        special_tokens=["<PAD>"],
    )
    texts = ["aa", "", "b", "a<PAD>a"]

    expected: list[int] = []
    for text in texts:
        expected.extend(tk.encode(text))

    assert list(tk.encode_iterable(texts)) == expected


@pytest.mark.unit
def test_encode_iterable_returns_iterator_and_is_lazy():
    tk = tokenizer(vocab=_base_vocab(), merges=[], special_tokens=[])
    consumed: list[str] = []

    def text_stream():
        consumed.append("start")
        yield "ab"
        consumed.append("after-first-yield")
        yield "cd"

    token_iter = tk.encode_iterable(text_stream())

    assert iter(token_iter) is token_iter
    assert consumed == []

    assert next(token_iter) == ord("a")
    assert consumed == ["start"]

    assert list(token_iter) == [ord("b"), ord("c"), ord("d")]
    assert consumed == ["start", "after-first-yield"]


@pytest.mark.unit
def test_encode_iterable_delays_source_exceptions_until_iteration():
    tk = tokenizer(vocab=_base_vocab(), merges=[], special_tokens=[])

    def text_stream():
        yield "ab"
        raise RuntimeError("boom")

    token_iter = tk.encode_iterable(text_stream())

    assert list(next(token_iter) for _ in range(2)) == [ord("a"), ord("b")]
    with pytest.raises(RuntimeError, match="boom"):
        next(token_iter)


@pytest.mark.unit
def test_encode_iterable_returns_empty_iterator_for_empty_input():
    tk = tokenizer(vocab=_base_vocab(), merges=[], special_tokens=[])

    assert list(tk.encode_iterable([])) == []


def _write_tokenizer_files(tmp_path, vocab_lines: list[str], merge_lines: list[str]):
    vocab_path = tmp_path / "vocab.txt"
    merge_path = tmp_path / "merges.txt"
    vocab_content = "\n".join(vocab_lines)
    merge_content = "\n".join(merge_lines)
    if vocab_content:
        vocab_content += "\n"
    if merge_content:
        merge_content += "\n"
    vocab_path.write_text(vocab_content, encoding="utf-8")
    merge_path.write_text(merge_content, encoding="utf-8")
    return vocab_path, merge_path


@pytest.mark.unit
def test_from_files_loads_vocab_merges_and_special_tokens(tmp_path):
    vocab_path, merge_path = _write_tokenizer_files(
        tmp_path,
        vocab_lines=["97\t61", "98\t62", "256\t6162"],
        merge_lines=["61\t62"],
    )

    tk = tokenizer.from_files(
        str(vocab_path),
        str(merge_path),
        special_tokens=["<PAD>"],
    )

    assert isinstance(tk, tokenizer)
    assert tk.vocab == {97: b"a", 98: b"b", 256: b"ab"}
    assert tk.merges == [(b"a", b"b")]
    assert tk.special_tokens == ["<PAD>"]


@pytest.mark.unit
def test_from_files_defaults_special_tokens_to_empty_list(tmp_path):
    vocab_path, merge_path = _write_tokenizer_files(
        tmp_path,
        vocab_lines=["97\t61"],
        merge_lines=[],
    )

    tk = tokenizer.from_files(str(vocab_path), str(merge_path))

    assert tk.special_tokens == []


@pytest.mark.unit
def test_decode_returns_empty_string_for_empty_ids():
    tk = tokenizer(vocab=_base_vocab(), merges=[], special_tokens=[])

    assert tk.decode([]) == ""


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "abc",
        " hi",
        "a\nb",
        "你好",
        "🙂",
        "a<PAD>a",
    ],
)
def test_decode_without_merges_reconstructs_utf8_text(text):
    tk = tokenizer(vocab=_base_vocab(), merges=[], special_tokens=["<PAD>"])

    assert tk.decode(_utf8_token_ids(text)) == text


@pytest.mark.unit
def test_decode_accepts_mixed_byte_and_merged_tokens():
    tk = tokenizer(
        vocab=_vocab_with_extra_tokens(b"ab", "你好".encode("utf-8")),
        merges=[],
        special_tokens=[],
    )

    assert tk.decode([256, 99, 257]) == "abc你好"


@pytest.mark.unit
def test_decode_joins_bytes_across_token_boundaries_before_utf8_decoding():
    tk = tokenizer(
        vocab=_vocab_with_extra_tokens(b"\xe4\xbd", b"\xa0\xe5\xa5\xbd"),
        merges=[],
        special_tokens=[],
    )

    assert tk.decode([256, 257]) == "你好"


@pytest.mark.unit
def test_decode_round_trips_encoded_text_with_merges_and_special_token_text():
    tk = tokenizer(
        vocab=_vocab_with_extra_tokens(b"aa"),
        merges=[(b"a", b"a")],
        special_tokens=["<PAD>"],
    )
    text = "aa<PAD>aaa"

    assert tk.decode(tk.encode(text)) == text


@pytest.mark.unit
def test_decode_raises_key_error_for_unknown_token_id():
    tk = tokenizer(vocab=_base_vocab(), merges=[], special_tokens=[])

    with pytest.raises(KeyError, match="999"):
        tk.decode([999])
