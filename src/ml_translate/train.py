import torch
from torch import Tensor
import re
from ml_translate.transformer import Transformer


class CharacterTokenizer:
    def __init__(self, d_seq):
        self.next_available_index: int = 1
        self.ctoi = {"<PAD>": 0}
        self.itoc = {0: "<PAD>"}
        self.d_seq = d_seq

    def ingest(self, text: str):
        chars = set(text)
        for ch in chars:
            if ch not in self.ctoi:
                index = self.next_available_index
                self.next_available_index += 1
                self.ctoi[ch] = index
                self.itoc[index] = ch

    def encode(self, text: str) -> Tensor:
        t = torch.tensor([self.ctoi[ch] for ch in text])
        return torch.nn.functional.pad(t, (0, self.d_seq - t.size(0)))

    def decode(self, tokens: Tensor) -> str:
        return "".join(self.itoc[int(i.item())] for i in tokens)


with open("data/eng-fra.txt") as file:
    d_seq = 100
    tokenizer = CharacterTokenizer(d_seq=d_seq)
    model = Transformer(d_model=512, d_seq=100, n_vocab=100)

    count = 0
    max_lines = 100
    for line in file:
        if count >= max_lines:
            break
        clean_line = line.strip()
        # Replaces NNBSP, which is used before punctuation in french
        clean_line = re.sub(r"\u202f", "", clean_line)
        parts = clean_line.split("\t", maxsplit=1)
        for part in parts:
            tokenizer.ingest(part)
        tokenized = [tokenizer.encode(part) for part in parts]
        model.forward(tokenized[0], tokenized[1])
        count += 1
