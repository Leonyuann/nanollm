"""Encode and decode text with a trained byte-level BPE tokenizer.

This module defines the runtime tokenizer used after BPE training has produced
a vocabulary and ordered merge list. The tokenizer applies GPT-style
pretokenization, preserves configured special tokens as hard boundaries, merges
byte pairs according to the trained merge order, and converts between text and
token IDs.

Training utilities for producing vocabularies and merge lists are implemented
in ``train_bpe.py``.

Typical usage example:
    tk = tokenizer.from_files(
        "outputs/tokenizer/vocab.txt",
        "outputs/tokenizer/merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    token_ids = tk.encode("hello world")
    text = tk.decode(token_ids)

Classes:
    tokenizer: Runtime byte-level BPE tokenizer that supports encoding,
        iterable encoding, decoding, and loading artifacts from files.

Methods:
    tokenizer.from_files: Create a tokenizer from serialized vocabulary and
        merge files.
    tokenizer.encode: Encode one text string into token IDs.
    tokenizer.encode_iterable: Lazily encode an iterable of text strings.
    tokenizer.decode: Decode token IDs back into text.
    tokenizer.encodevocab: Build the reverse vocabulary from byte tokens to
        token IDs.

Notes:
    The vocabulary file should contain one token ID and byte representation per
    line, separated by a tab character. The merge file should contain one byte
    pair per line, also separated by a tab character.
"""


import re
from typing import TypeAlias
from collections.abc import Iterable, Iterator
from tokenizer.types import Vocabulary, Merges
from tokenizer.train_bpe import pretokenize


EncodeVocabulary: TypeAlias = dict[bytes, int]


class tokenizer:
    """
    A tokenizer class that encodes and decodes text using a given vocabulary and merges.
    It also handles special tokens.

    Attributes:
        vocab: A dictionary mapping token IDs to their corresponding byte representations.
        merges: A list of byte pairs that should be merged during encoding.
        special_tokens: A list of special tokens that should be treated as hard boundaries during 
        pretokenization.
        merge_ranks: A dictionary mapping byte pairs to their rank in the merges list, used for
        quickly finding the order of merges.
        encode_vocab: A dictionary mapping byte tokens to their corresponding token IDs, used for
        encoding text into token IDs.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        merges: Merges,
        special_tokens: list[str] = None,
    ):
        self.vocab = vocab
        self.merges = merges

        self.special_tokens = special_tokens or []

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(merges)
        }

        self.encode_vocab ={
            token: idx 
            for idx, token in self.vocab.items()
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merge_filepath: str,
        special_tokens: list[str] = None,
    ):
        """
        Create a tokenizer instance from vocabulary and merges files.

        Args:
            vocab_filepath: The file path to the vocabulary file, which should contain token IDs 
            and their corresponding byte representations.
            merge_filepath: The file path to the merges file, which should contain byte pairs that
            should be merged during encoding.
            special_tokens: A list of special tokens that should be treated as hard boundaries 
            during pretokenization.

        Returns:
            An instance of the tokenizer class initialized with the provided vocabulary, merges, 
            and special tokens.
        """
        vocab: Vocabulary = {}
        merges: Merges = []

        with open(vocab_filepath, "r", encoding="utf-8") as vocab_file:
            for line in vocab_file:
                token_id_str, byte_str = line.strip().split("\t")
                token_id = int(token_id_str)
                byte_token = bytes.fromhex(byte_str)
                vocab[token_id] = byte_token

        with open(merge_filepath, "r", encoding="utf-8") as merge_file:
            for line in merge_file:
                byte1_str, byte2_str = line.strip().split("\t")
                byte1 = bytes.fromhex(byte1_str)
                byte2 = bytes.fromhex(byte2_str)
                merges.append((byte1, byte2))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs in vocab.

        Args:
            text: The input string to encode.

        Returns:
            A list of token IDs corresponding to the input string.
        """
        text_encoded: list[int] = []
        special_token_set = set(self.special_tokens)

        for segment in self._split_text_with_special_tokens(text):
            if segment in special_token_set:
                text_encoded.append(self.encode_vocab[segment.encode("utf-8")])
                continue

            words = pretokenize(segment, [])

            # For each pre-tokenized word, encode it into bytes and apply merges to get the
            # final token IDs.
            for word in words:
                token: list[int] = []
                
                bytes_word = self._encode_a_word(word)

                # After applying all merges, convert the final byte tokens to their corresponding
                # token IDs in the vocab.
                for byte_token in bytes_word:
                    token.append(self.encode_vocab[byte_token])

                text_encoded = text_encoded + token

        return text_encoded

    def _split_text_with_special_tokens(
        self,
        text: str,
    ) -> list[str]:
        """
        Split text while preserving configured special tokens.

        Args:
            text: The input string to split.

        Returns:
            A list of normal text segments and special token segments.
        """
        if not self.special_tokens:
            return [text]

        pattern = "|".join(
            re.escape(token)
            for token in sorted(self.special_tokens, key=len, reverse=True)
        )
        return [segment for segment in re.split(f"({pattern})", text) if segment]

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.

        Args:
            iterable: An iterable of strings to encode.

        Returns:
            A generator that yields token IDs corresponding to the input strings.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.

        Args:
            ids: A list of token IDs to decode.

        Returns:
            The decoded string corresponding to the input token IDs.
        """
        decode_bytes: bytes = b""
        for token in ids:
            decode_bytes += self.vocab[token]
        return decode_bytes.decode("utf-8", errors="replace")
    
    def _encode_a_word (
        self,
        word: str,
    ) -> tuple[int]:
        bytes_word = tuple(bytes([byte]) for byte in word.encode("utf-8"))

        while len(bytes_word) > 1:
            prior_pair = None
            prior_rank = float("inf")

            for pair in zip(bytes_word, bytes_word[1:]):
                rank = self.merge_ranks.get(pair)

                if rank is not None and rank < prior_rank:
                    prior_rank = rank
                    prior_pair = pair

            if prior_pair is None:
                break

            bytes_word = self._pair_merge(prior_pair, bytes_word)

        return bytes_word

            
    def _pair_merge(
        self,
        merge: tuple[bytes, bytes],
        word: tuple[bytes],
    ) -> tuple[bytes]:
        # If the word has fewer than 2 byte tokens, we can't merge any more pairs,
        # so we break out
        lenth = len(word)
        if lenth < 2:
            return word

        i = 0
        while i < lenth - 1:
            if word[i] != merge[0]:
                i += 1
                continue
            if word[i + 1] != merge[1]:
                i += 1
                continue

            # If the merge matches, replace the two tokens with the merged token.
            merged_token: bytes = merge[0] + merge[1]
            word = (word[:i] + (merged_token,) + word[i + 2 :])

            lenth -= 1
            i += 1

        return word
                    