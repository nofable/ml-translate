from __future__ import unicode_literals, print_function, division

import re
import unicodedata

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from ml_translate.utils import get_project_root

MAX_LENGTH: int = 10

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
    print("Reading lines...")
    project_root = get_project_root()

    # Read the file and split into lines
    lines = (
        open(project_root / f"data/{lang1}-{lang2}.txt", encoding="utf-8")
        .read()
        .strip()
        .split("\n")
    )

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
    print(f"Read {len(pairs)} sentence pairs")
    pairs = filterPairs(pairs)
    print(f"Trimmed to {len(pairs)} sentence pairs")
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

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
    return [lang.word2index[word] for word in sentence.split(" ")]


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
    batch_size: int, device: torch.device
) -> tuple[Lang, Lang, list[list[str]], DataLoader[tuple[Tensor, ...]]]:
    input_lang, output_lang, pairs = readLangs("eng", "fra", True)

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
