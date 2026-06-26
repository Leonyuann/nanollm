"""Deduplication functions for processing raw data.
"""
from pathlib import Path
import hashlib

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
