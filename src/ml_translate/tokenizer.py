import torch
from torch import Tensor


class CharacterTokenizer:
    def __init__(self, seq_len):
        self.next_available_index: int = 3
        self.ctoi = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
        self.itoc = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
        self.seq_len = seq_len

    def ingest(self, text: str):
        chars = set(text)
        for ch in chars:
            if ch not in self.ctoi:
                index = self.next_available_index
                self.next_available_index += 1
                self.ctoi[ch] = index
                self.itoc[index] = ch

    def encode(self, text: str, right_shift: bool = False) -> Tensor:
        proper = [self.ctoi[ch] for ch in text]
        t: Tensor
        if right_shift:
            t = torch.tensor([1] + proper + [2])
        else:
            t = torch.tensor(proper + [2])

        return torch.nn.functional.pad(t, (0, self.seq_len - t.size(0)))

    def decode(self, tokens: Tensor) -> str:
        result = ""
        for i in tokens:
            key = int(i.item())
            if key in self.itoc:
                result += self.itoc[key]
            else:
                result += "<UNK>"
        return result
