import logging
import re
import unicodedata
from typing import Callable

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

from ml_translate.config import default_config
from ml_translate.utils import get_project_root

logger = logging.getLogger(__name__)

MAX_LENGTH: int = default_config.max_length

eng_prefixes: tuple[str, ...] = (
    "i am ",
    "i m ",
    "he is",
    "he s ",
    "she is",
    "she s ",
    "you are",
    "you re ",
    "we are",
    "we re ",
    "they are",
    "they re ",
)

SOS_token: int = 0
EOS_token: int = 1


class Lang:
    def __init__(self, name: str) -> None:
        self.name = name
        self.word2index: dict[str, int] = {}
        self.word2count: dict[str, int] = {}
        self.index2word: dict[int, str] = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence: str) -> None:
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word: str) -> None:
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s: str) -> str:
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def readLangs(
    lang1: str, lang2: str, reverse: bool = False
) -> tuple[Lang, Lang, list[list[str]]]:
    logger.info("Reading lines...")
    project_root = get_project_root()

    # Check if data file exists
    data_path = project_root / f"data/{lang1}-{lang2}.txt"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            f"Please ensure the {lang1}-{lang2}.txt file exists in the data directory."
        )

    # Read the file and split into lines
    with open(data_path, encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in line.split("\t")] for line in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    # Apply filtering and add sentences to the Lang objects
    logger.info("Read %d sentence pairs", len(pairs))
    pairs = filterPairs(pairs)
    logger.info("Trimmed to %d sentence pairs", len(pairs))
    logger.info("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    logger.info("Counted words:")
    logger.info("%s: %d", input_lang.name, input_lang.n_words)
    logger.info("%s: %d", output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs


def filterPair(p: list[str]) -> bool:
    return (
        len(p[0].split(" ")) < MAX_LENGTH and len(p[1].split(" ")) < MAX_LENGTH
        # and p[1].startswith(eng_prefixes) # uncomment this line for faster training on cpu hardware
    )


def filterPairs(pairs: list[list[str]]) -> list[list[str]]:
    return [pair for pair in pairs if filterPair(pair)]


def split_pairs(
    pairs: list[list[str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Split pairs into train, validation, and test sets using sklearn.

    Args:
        pairs: List of sentence pairs to split.
        train_ratio: Fraction of data for training (default 0.8).
        val_ratio: Fraction of data for validation (default 0.1).
        test_ratio: Fraction of data for testing (default 0.1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs).
    """
    if len(pairs) == 0:
        return [], [], []

    # First split: train vs (val + test)
    train_pairs, temp_pairs = train_test_split(
        pairs, train_size=train_ratio, random_state=seed
    )

    # Second split: val vs test
    if val_ratio == 0:
        val_pairs, test_pairs = [], temp_pairs
    elif test_ratio == 0:
        val_pairs, test_pairs = temp_pairs, []
    else:
        val_size = val_ratio / (val_ratio + test_ratio)
        val_pairs, test_pairs = train_test_split(
            temp_pairs, train_size=val_size, random_state=seed
        )

    logger.info(
        "Split %d pairs into train=%d, val=%d, test=%d",
        len(pairs),
        len(train_pairs),
        len(val_pairs),
        len(test_pairs),
    )

    return train_pairs, val_pairs, test_pairs


def indexesFromSentence(lang: Lang, sentence: str) -> list[int]:
    indexes: list[int] = []
    for word in sentence.split(" "):
        if word not in lang.word2index:
            raise ValueError(
                f"Unknown word '{word}' not found in vocabulary for language '{lang.name}'"
            )
        indexes.append(lang.word2index[word])
    return indexes


def tensorFromSentence(lang: Lang, sentence: str, device: torch.device) -> Tensor:
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def tensorsFromPair(
    pair: list[str], input_lang: Lang, output_lang: Lang, device: torch.device
) -> tuple[Tensor, Tensor]:
    input_tensor = tensorFromSentence(input_lang, pair[0], device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device)
    return (input_tensor, target_tensor)


class TranslationDataset(Dataset):
    """Dataset that stores sentence pairs as token indices for dynamic batching."""

    def __init__(
        self,
        pairs: list[list[str]],
        input_lang: Lang,
        output_lang: Lang,
    ):
        """
        Args:
            pairs: List of [input_sentence, output_sentence] pairs.
            input_lang: Language object for input vocabulary.
            output_lang: Language object for output vocabulary.
        """
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        inp, tgt = self.pairs[idx]
        inp_ids = indexesFromSentence(self.input_lang, inp)
        tgt_ids = indexesFromSentence(self.output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        return inp_ids, tgt_ids


def collate_dynamic_batch(
    batch: list[tuple[list[int], list[int]]], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Collate function that pads sequences to the max length in the batch.

    Args:
        batch: List of (input_ids, target_ids) tuples.
        device: Device to place tensors on.

    Returns:
        Tuple of (input_tensor, target_tensor) padded to batch max lengths.
    """
    input_seqs, target_seqs = zip(*batch)

    # Convert to tensors
    input_tensors = [torch.tensor(seq, dtype=torch.long) for seq in input_seqs]
    target_tensors = [torch.tensor(seq, dtype=torch.long) for seq in target_seqs]

    # Pad sequences to max length in batch (padding value = 0)
    input_padded = pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target_tensors, batch_first=True, padding_value=0)

    return input_padded.to(device), target_padded.to(device)


def _create_tensor_dataset(
    pairs: list[list[str]],
    input_lang: Lang,
    output_lang: Lang,
    device: torch.device,
) -> TensorDataset:
    """Convert sentence pairs to a TensorDataset.

    Args:
        pairs: List of [input_sentence, output_sentence] pairs.
        input_lang: Language object for input vocabulary.
        output_lang: Language object for output vocabulary.
        device: Device to place tensors on.

    Returns:
        TensorDataset with input and target tensors.
    """
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, : len(inp_ids)] = inp_ids
        target_ids[idx, : len(tgt_ids)] = tgt_ids

    return TensorDataset(
        torch.LongTensor(input_ids).to(device),
        torch.LongTensor(target_ids).to(device),
    )


def get_dataloaders(
    batch_size: int,
    device: torch.device,
    lang1: str = "eng",
    lang2: str = "fra",
    reverse: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    dynamic_batching: bool = False,
) -> tuple[
    Lang,
    Lang,
    DataLoader[tuple[Tensor, ...]],
    DataLoader[tuple[Tensor, ...]],
    DataLoader[tuple[Tensor, ...]],
    list[list[str]],
]:
    """Load data and create train/val/test dataloaders.

    Args:
        batch_size: Number of samples per batch.
        device: Device to place tensors on.
        lang1: First language code (default "eng").
        lang2: Second language code (default "fra").
        reverse: If True, reverse translation direction (default True).
        train_ratio: Fraction of data for training (default 0.8).
        val_ratio: Fraction of data for validation (default 0.1).
        test_ratio: Fraction of data for testing (default 0.1).
        seed: Random seed for reproducible splits.
        dynamic_batching: If True, pad sequences to batch max length instead
            of global MAX_LENGTH. Reduces memory and speeds up training.

    Returns:
        Tuple of (input_lang, output_lang, train_loader, val_loader, test_loader, test_pairs).
        test_pairs is included for use with evaluateRandomly().
    """
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)

    # Split the data
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs, train_ratio, val_ratio, test_ratio, seed
    )

    if dynamic_batching:
        # Use dynamic batching with custom collate function
        train_data = TranslationDataset(train_pairs, input_lang, output_lang)
        val_data = TranslationDataset(val_pairs, input_lang, output_lang)
        test_data = TranslationDataset(test_pairs, input_lang, output_lang)

        collate_fn: Callable = lambda batch: collate_dynamic_batch(batch, device)

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:
        # Use fixed padding to MAX_LENGTH (original behavior)
        train_data = _create_tensor_dataset(
            train_pairs, input_lang, output_lang, device
        )
        val_data = _create_tensor_dataset(val_pairs, input_lang, output_lang, device)
        test_data = _create_tensor_dataset(test_pairs, input_lang, output_lang, device)

        train_loader = DataLoader(
            train_data,
            sampler=RandomSampler(train_data),
            batch_size=batch_size,
        )
        val_loader = DataLoader(
            val_data,
            sampler=SequentialSampler(val_data),
            batch_size=batch_size,
        )
        test_loader = DataLoader(
            test_data,
            sampler=SequentialSampler(test_data),
            batch_size=batch_size,
        )

    return input_lang, output_lang, train_loader, val_loader, test_loader, test_pairs
