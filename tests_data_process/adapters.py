from __future__ import annotations

import os
from typing import Any

from data_process.filter import mask_emails, mask_phone_numbers, mask_ips, language_identificatin, gopher_quality_filter
from data_process.deduplication import exact_deduplication_on_files, minhash_deduplicatin

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    raise NotImplementedError


def run_identify_language(text: str) -> tuple[Any, float]:
    return language_identificatin(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    return gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return exact_deduplication_on_files(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    return minhash_deduplicatin(
        input_files=input_files,
        num_hash_fun=num_hashes,
        num_bands=num_bands,
        n_gram_length=ngrams,
        theshold=jaccard_threshold,
        output_dir=output_directory,
    )
