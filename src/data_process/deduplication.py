"""Deduplication functions for processing raw data.
"""
from pathlib import Path
import hashlib

from datasketch import MinHash, MinHashLSH
import nltk 

class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.size = [1 for _ in range(size)]

    def find(self, x) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, a: int, b: int) -> bool:
        root_a = self.parent[a]
        root_b = self.parent[b]

        if root_a == root_b:
            return False
        
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_b

        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]
        return True
        

def exact_deduplication_on_files(input_files: list[str], output_dir: str):
    """Remove duplicate lines from a list of files and write the unique lines to a new directory.
    
    Args:
        input_files: List of file paths to process.
        output_dir: Directory path where deduplicated files will be written.        

    Returns:
        None. Deduplicated files are written to the specified output directory.  
    """

    for input_file in input_files:
        if Path(input_file).exists() is False:
            raise FileExistsError(f"exact_deduplication_on_files(): {input_file} not exists.")
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    repeated_num = {}
    # Count repeated sentences among all files
    for input_file in input_files:
        with open(input_file, mode="rb") as source:
            for line in source:
                h = hashlib.sha256(line)
                repeated_num[h.digest()] = repeated_num.get(h.digest(),0) + 1

    # Deduplicate for all files, and write into new files.
    for input_file in input_files:
        target_file = output_dir / Path(input_file).name
        with (open(input_file, mode="rb") as source,
            open(target_file, mode="wb") as target):        
            for line in source:
                if repeated_num[hashlib.sha256(line).digest()] > 1:
                    continue

                target.write(line)

def minhash_deduplicatin(
    input_files: list[str],
    num_hash_fun: int,
    num_bands: int,
    n_gram_length: int,
    output_dir: str,
    theshold: float = 0.9,
):
    """ Remove near-duplicate files from a list of files using MinHash and 
    Locality-Sensitive Hashing (LSH) and write the unique files to a new directory.
    
    Args:
        input_files: List of file paths to process.
        num_hash_fun: Number of hash functions to use for MinHash.
        num_bands: Number of bands to use for LSH.      
        n_gram_length: Length of n-grams to use for MinHash.
        output_dir: Directory path where deduplicated files will be written.
        theshold: Jaccard similarity threshold for considering files as duplicates. Default is 0.9.

    """

    # Check if all input files are exist
    for input_file in input_files:
        if Path(input_file).exists() is False:
            raise FileNotFoundError(f"minhash_deduplicatin(): {input_file} doesn't exist.")
        
    # Check if the hypherparameters is valid
    if num_hash_fun <= 0:
        raise ValueError(f"minhash_deduplicatin(): number of hash functions is invalid.")
    if num_bands <= 0:
        raise ValueError(f"minhash_deduplicatin(): number of bands is invalid.")
    if n_gram_length <= 0:
        raise ValueError(f"minhash_deduplicatin(): length of n-gram is invalid.")
    if num_hash_fun % num_bands != 0:
        raise ValueError(f"minhash_deduplicatin(): number of hash function should be divisible by number of bands.")

    # Initialize output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lsh = MinHashLSH(
        threshold=theshold, 
        num_perm=num_hash_fun,
        params=(num_bands, num_hash_fun // num_bands),
    )

    minhashes: list = {}
    # Apply LSH
    for file_id, minhash in enumerate(__make_minhash(input_files, num_hash_fun, n_gram_length)):
        minhashes[file_id] = minhash
        lsh.insert(file_id, minhash)

    uf_set = UnionFind(len(input_files))    

    # Foe each file, computing Jaccab similarity with its candidates to determine 
    # wheather to treat them in a cluster 
    for file_id in range(len(input_files)):
        # For each candicate 
        for candidate in lsh.query(minhashes[file_id]):
            if candidate == file_id:
                continue

            if minhashes[file_id].jaccard(minhashes[candidate]) < theshold:
                continue
            uf_set.union(file_id, candidate)

    kept_roots: set[int] = set()

    for file_id, input_file in enumerate(input_files):
        root = uf_set.find(file_id)
        if root in kept_roots:
            continue

        kept_roots.add(root)

        source_file = Path(input_file)
        output_file = output_dir / source_file.name
        output_file.write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")


def __make_minhash(input_files: str, num_hash_fun: int, n_gram_length: int) :
    for input_file in input_files:
        text = Path(input_file).read_text(encoding="utf-8")
        words = nltk.word_tokenize(text)

        if len(words) < n_gram_length:
            n_grams = [" ".join(words)] if words else [""]
        else:
            n_grams = [" ".join(words[i:i + n_gram_length]) for i in range(len(words) - n_gram_length + 1)]

        minhash = MinHash(num_perm=num_hash_fun)
        for n_gram in n_grams:
            minhash.update(n_gram.encode("utf-8"))

        yield minhash

