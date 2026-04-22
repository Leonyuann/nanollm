from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, IterableDataset

from config_manager import DataConfig

if TYPE_CHECKING:
    from tokenizer.tokenizer import tokenizer

    from .config import TrainingConfig


class TokenSequenceIterableDataset(IterableDataset[torch.Tensor]):
    """Stream fixed-length token windows from a text file.

    Extended description.

    Attributes:
        file_path: Source text file.
        text_tokenizer: Runtime tokenizer used to turn text into token ids.
        seq_len: Input sequence length expected by the language model.
        repeat: Whether iteration restarts from the beginning after EOF.
        window_size: Number of tokens yielded per sample, including targets.

    Notes:
        Each yielded sample has length ``seq_len + 1`` so the caller can split
        it into shifted ``input_ids`` and ``targets``.
    """

    def __init__(
        self,
        file_path: str,
        text_tokenizer: tokenizer,
        seq_len: int,
        repeat: bool,
    ) -> None:
        """Initialize the streaming dataset.

        Args:
            file_path: Text corpus path.
            text_tokenizer: Tokenizer instance used for encoding.
            seq_len: Model input sequence length.
            repeat: Whether to keep looping over the file.

        Returns:
            None.

        Raises:
            ValueError: If ``seq_len`` is not positive.
        """
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        self.file_path = Path(file_path)
        self.text_tokenizer = text_tokenizer
        self.seq_len = seq_len
        self.repeat = repeat
        self.window_size = seq_len + 1

    def _iterate_lines(self):
        """Stream text lines from the source file for one pass.

        Args:
            None.

        Returns:
            An iterator over lines, preserving newline characters.
        """
        with self.file_path.open("r", encoding="utf-8") as source_file:
            yield from source_file

    def __iter__(self):
        """Yield contiguous fixed-length token windows from the source file.

        Args:
            None.

        Returns:
            An iterator over ``torch.long`` tensors with shape ``[seq_len + 1]``.

        Raises:
            ValueError: If a full dataset pass cannot produce even one window.
        """
        while True:
            token_buffer: list[int] = []
            yielded_window = False

            for token_id in self.text_tokenizer.encode_iterable(self._iterate_lines()):
                token_buffer.append(token_id)
                if len(token_buffer) < self.window_size:
                    continue

                yielded_window = True
                yield torch.tensor(token_buffer, dtype=torch.long)
                token_buffer = []

            if not yielded_window:
                raise ValueError(
                    f"{self.file_path} does not contain enough tokens to form a "
                    f"single window of length {self.window_size}"
                )
            if not self.repeat:
                return


def resolve_dataset_paths(
    data_config: DataConfig,
    dataset_name: str,
) -> tuple[str, str]:
    """Map a dataset alias to its train and validation files.

    Args:
        data_config: Repository data-path configuration.
        dataset_name: Requested dataset alias.

    Returns:
        A ``(train_path, valid_path)`` tuple.

    Raises:
        ValueError: If ``dataset_name`` is not supported.
    """
    normalized_name = dataset_name.lower()
    if normalized_name == "tinystories":
        return data_config.TinyStories_train_path, data_config.TinyStories_valid_path
    if normalized_name == "owt":
        return data_config.owt_train_path, data_config.owt_valid_path
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_dataloaders(
    data_config: DataConfig,
    training_config: TrainingConfig,
    text_tokenizer: tokenizer,
    seq_len: int,
) -> tuple[DataLoader, DataLoader]:
    """Construct training and validation dataloaders.

    Args:
        data_config: Repository data-path configuration.
        training_config: Training loop configuration.
        text_tokenizer: Tokenizer instance used for encoding.
        seq_len: Input sequence length expected by the model.

    Returns:
        A tuple ``(train_loader, valid_loader)``.
    """
    train_path, valid_path = resolve_dataset_paths(data_config, training_config.dataset)
    train_dataset = TokenSequenceIterableDataset(
        file_path=train_path,
        text_tokenizer=text_tokenizer,
        seq_len=seq_len,
        repeat=True,
    )
    valid_dataset = TokenSequenceIterableDataset(
        file_path=valid_path,
        text_tokenizer=text_tokenizer,
        seq_len=seq_len,
        repeat=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=training_config.batch_size,
        num_workers=0,
    )
    return train_loader, valid_loader
