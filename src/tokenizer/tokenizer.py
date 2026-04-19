from tokenizer.types import Vocabulary, Merges
from tokenizer.train_bpe import pretokenize
from config_manager import load_BPEConfig
from typing import TypeAlias
from collections.abc import Iterable, Iterator

EncodeVocabulary: TypeAlias = dict[bytes, int]  

class tokenizer:
    """
    A tokenizer class that encodes and decodes text using a given vocabulary and merges. It also handles special tokens.

    Attributes:
        vocab: A dictionary mapping token IDs to their corresponding byte representations.
        merges: A list of byte pairs that should be merged during encoding.
        special_tokens: A list of special tokens that should be treated as hard boundaries during pretoken
        ization.
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
            vocab_filepath: The file path to the vocabulary file, which should contain token IDs and their corresponding byte representations.
            merge_filepath: The file path to the merges file, which should contain byte pairs that should be merged during encoding.
            special_tokens: A list of special tokens that should be treated as hard boundaries during pretokenization.
        
        Returns:
            An instance of the tokenizer class initialized with the provided vocabulary, merges, and special tokens.
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
    
    def encode(
        self,
        text: str
    ) -> list[int]: 
        """
        Encode a string into a list of token IDs in vocab.

        Args:
            text: The input string to encode.

        Returns:
            A list of token IDs corresponding to the input string.
        """   
        # Pre-tokenize the text, removing special tokens and building a pre-token list.
        envocab = self.encodevocab()
        text_encoded : list[int]= []
        bpeconfig = load_BPEConfig()
        words = pretokenize(text, bpeconfig.special_tokens)

        # For each pre-tokenized word, encode it into bytes and apply merges to get the final token IDs.
        for word in words:
            bytes_word = tuple(bytes([byte]) for byte in word.encode("utf-8"))
            token : list[int]= []

            # For each byte-pair in the word, merge it according to the merges list..
            for merge in self.merges:
                # If the word has fewer than 2 byte tokens, we can't merge any more pairs, so we break out of the loop.
                lenth = len(bytes_word)
                if lenth < 2:
                    break

                i = 0
                while i < lenth - 1:
                    if bytes_word[i] != merge[0]:
                        i += 1
                        continue
                    if bytes_word[i + 1] != merge[1]:
                        i += 1
                        continue

                    # If the merge matches, replace the two tokens with the merged token.
                    merged_token: bytes = merge[0] + merge[1]
                    bytes_word = bytes_word[:i] + (merged_token,) + bytes_word[i + 2:]

                    lenth -= 1
                    i += 1
                    
            # After applying all merges, convert the final byte tokens to their corresponding token IDs in the vocab.
            for byte_token in bytes_word:
                token.append(envocab[byte_token])
            
            text_encoded = text_encoded + token

        return text_encoded
    
    
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


    def decode(
        self,
        ids: list[int]
    ) -> str:
        """
        Decode a sequence of token IDs into text.

        Args:
            ids: A list of token IDs to decode.

        Returns:
            The decoded string corresponding to the input token IDs.
        """
        decode_bytes : bytes = b""
        for token in ids:
            decode_bytes += self.vocab[token]
        return decode_bytes.decode("utf-8", errors="replace")

    def encodevocab(
        self,
    ) -> EncodeVocabulary:
        return {token: idx for idx, token in self.vocab.items()}
