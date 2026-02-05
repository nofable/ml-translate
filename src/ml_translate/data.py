import logging
import re
import unicodedata

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

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
        len(p[0].split(" ")) < MAX_LENGTH
        and len(p[1].split(" ")) < MAX_LENGTH
        and p[1].startswith(eng_prefixes)
    )


def filterPairs(pairs: list[list[str]]) -> list[list[str]]:
    return [pair for pair in pairs if filterPair(pair)]


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


def get_dataloader(
    batch_size: int,
    device: torch.device,
    lang1: str = "eng",
    lang2: str = "fra",
    reverse: bool = True,
) -> tuple[Lang, Lang, list[list[str]], DataLoader[tuple[Tensor, ...]]]:
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)

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

    train_data = TensorDataset(
        torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device)
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size
    )
    return input_lang, output_lang, pairs, train_dataloader
