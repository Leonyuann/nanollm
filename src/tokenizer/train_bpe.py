import os
import regex as re
from typing import TypeAlias, BinaryIO
from multiprocessing import Pool
from collections import Counter, defaultdict
from config_manager import BPEConfig, load_BPEConfig
from tokenizer.types import Vocabulary, Merges

FrequencyTable: TypeAlias = dict[tuple[bytes, ...], int]
PairCounts: TypeAlias = dict[tuple[bytes, bytes], int]
PairLocations: TypeAlias = dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
WordPairs: TypeAlias = dict[tuple[bytes, ...], Counter[tuple[bytes, bytes]]]

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def uft8_vocab() -> Vocabulary:
    """
    Get a vocabulary of all single-byte UTF-8 characters.

    Returns:
        A dictionary mapping byte values (0-255) to their corresponding single-byte UTF-8 characters.
    """
    vocab = {i: bytes([i]) for i in range(256)}
    return vocab


def remove_special_tokens(
    text: str,
    special_tokens: list[str],
) -> list[str]:
    """
    Remove special tokens from the text.

    Args:
        text: The input text to be processed.
        special_tokens: A list of special tokens to be removed from the text.
    
    Returns:
        A list of strings with the special tokens removed.
    """
    if not special_tokens:
        return [text]
    pattern = "|".join(map(re.escape, special_tokens))
    return re.split(pattern, text)


def pretokenization(
    chunk: str,
    special_tokens: list[str],
) ->  FrequencyTable:
    """
    Pretokenize a chunk of the input file.

    Args:
        chunk: The chunk of text to be pretokenized.
        special_tokens: A list of special tokens acted as hard boundaries.
    
    Returns:
        A frequency table of token pairs in the chunk.
    """
    fre_table: FrequencyTable = {}
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    ## Remove special tokens
    chunks = remove_special_tokens(chunk, special_tokens)

    for chunk in chunks:
        words = re.finditer(PAT,chunk)
        for word in words:
            byte_words: bytes = word.group().encode("utf-8")
            tokens = tuple(bytes([byte]) for byte in byte_words)
            fre_table[tokens] = fre_table.get(tokens, 0) + 1

    return fre_table


def merge_initial(
    vocab: Vocabulary,
    fre_table: FrequencyTable,
) -> tuple[PairCounts, PairLocations]:
    """
    Count the frequency of each token pair in the frequency table and track their locations.

    Args:
    vocab: The current vocabulary of tokens.
    fre_table: The frequency table of token pairs.

    Returns:
    A tuple containing:
    - A dictionary mapping token pairs to their frequency counts.
    - A dictionary mapping token pairs to a list of their locations in the frequency table.
    """
    pair_counts: PairCounts = {}
    pair_locations: PairLocations = {}

    for word in fre_table:
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + fre_table[word]
            pair_locations[pair] = pair_locations.get(pair, []) + [word]
    return pair_counts, pair_locations


def count_pairs_in_word(
    word: tuple[bytes, ...],
) -> Counter[tuple[bytes, bytes]]:
    pair_counter: Counter[tuple[bytes, bytes]] = Counter()
    for i in range(len(word) - 1):
        pair_counter[(word[i], word[i + 1])] += 1
    return pair_counter


def merge_pair(
    pair: tuple[bytes, bytes],
    word: tuple[bytes, ...],
) -> tuple[bytes, ...]:
    """
    Merge a given pair of tokens in a word.

    Args:
        pair: The pair of tokens to be merged.
        word: The word in which the pair should be merged.

    Returns:
        A new word with the specified pair merged into a single token.
    """
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i+1]) == pair:
            new_word.append(b"".join(pair))
            i += 2  # Skip the next token since it's part of the merged pair
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def merge(
    vocab: Vocabulary,
    fre_table: FrequencyTable,
    vocab_size: int
) -> tuple[Vocabulary, Merges]:
    """
    Merge the most frequent token pairs until the desired vocabulary size is reached.

    Args:
    vocab: The current vocabulary of tokens.
    fre_table: The frequency table of token pairs.
    vocab_size: The desired size of the final vocabulary (including special tokens).

    Returns:
    A tuple containing:
    - The final vocabulary after merging.
    - A list of merges performed, where each merge is represented as a tuple of the merged token pair.
    """

    merges: Merges = []
    pair_counts: PairCounts = {}
    pair_locations: PairLocations = defaultdict(set)
    word_pairs: WordPairs = {}

    # Initialize pair counts and locations based on the initial frequency table
    for word, count in fre_table.items():
        pair_counter = count_pairs_in_word(word)
        word_pairs[word] = pair_counter
        for pair, occurrences in pair_counter.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + occurrences * count
            pair_locations[pair].add(word)

    while len(vocab) < vocab_size and pair_counts:
        # Find the most frequent pair, breaking ties by lexicographical order
        most_frequent_pair = max(pair_counts, key=lambda k: (pair_counts[k], k))
        if pair_counts[most_frequent_pair] <= 0:
            break
        
        # Get the list of words affected by the most frequent pair
        affected_words = list(pair_locations.get(most_frequent_pair, set()))
        if not affected_words:
            pair_counts.pop(most_frequent_pair, None)
            pair_locations.pop(most_frequent_pair, None)
            continue

        # Merge the most frequent pair
        merges.append(most_frequent_pair)
        vocab[len(vocab)] = b"".join(most_frequent_pair)

        # Update the frequency table and pair counts for the affected words
        for word in affected_words:
            count = fre_table.pop(word, 0)
            if count == 0:
                continue
            
            # Get the old pairs and new pairs for the word after merging
            old_pairs = word_pairs.pop(word)
            new_word = merge_pair(most_frequent_pair, word)
            new_pairs = word_pairs.get(new_word)
            if new_pairs is None:
                new_pairs = count_pairs_in_word(new_word)
                word_pairs[new_word] = new_pairs

            fre_table[new_word] = fre_table.get(new_word, 0) + count

            # Update pair locations and counts for the old pairs and new pairs
            for pair in old_pairs:
                locations = pair_locations.get(pair)
                if locations is not None:
                    locations.discard(word)
                    if not locations:
                        pair_locations.pop(pair, None)

            # Update locations and counts for the new pairs
            for pair in new_pairs:
                pair_locations[pair].add(new_word)

            # Update pair counts for all pairs that were affected by the merge
            for pair in old_pairs.keys() | new_pairs.keys():
                new_count = pair_counts.get(pair, 0) + (new_pairs.get(pair, 0) - old_pairs.get(pair, 0)) * count
                if new_count > 0:
                    pair_counts[pair] = new_count
                else:
                    pair_counts.pop(pair, None)

        # Remove the merged pair from the counts and locations
        pair_locations.pop(most_frequent_pair, None)
        pair_counts.pop(most_frequent_pair, None)

    return vocab, merges


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[Vocabulary, Merges]:
    """
    Train a BPE tokenizer on the given input file and special tokens.

    Args:
        input_path: The path to the input file to be tokenized.
        vocab_size: The desired size of the final vocabulary (including special tokens).
        special_tokens: A list of special tokens to be included in the vocabulary and treated as hard boundaries during tokenization.

    Returns:
        Final vocabulary and list of merges.
    """
    # Initialize with UTF-8 single-byte characters as the initial vocabulary
    config = load_BPEConfig()
    vocab = uft8_vocab()
    num_process = config.num_process

    # Add special tokens to the vocabulary
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    # Find chunks of the input file that can be processed independently, split on special token boundaries
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_process, split_special_token=b"<|endoftext|>")

        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end-start).decode("utf-8", errors="ignore"))

    # Pretokenize each chunk in parallel, counting the frequency of each token pair
    with Pool(num_process) as pool: 
        fre_tables = pool.starmap(pretokenization, [(chunk, special_tokens) for chunk in chunks])
    
    # Combine frequency tables from all chunks into a single frequency table
    fre_table: FrequencyTable = {}
    for word in fre_tables:
        for pair, count in word.items():
            fre_table[pair] = fre_table.get(pair, 0) + count

    vocab, merges = merge(vocab, fre_table, vocab_size)

    return vocab, merges
    
