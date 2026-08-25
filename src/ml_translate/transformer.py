import torch
from torch import Tensor
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, training_text: str, d_model: int, d_seq: int, n_vocab: int):
        super().__init__()
        self.tokenizer = Tokenizer(training_text)
        self.positionalEncoder = PositionalEncoder(d_model, d_seq)
        self.encoder_embed = nn.Embedding(n_vocab, d_model)
        self.decoder_embed = nn.Embedding(n_vocab, d_model)


class Tokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str):
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        return "".join(self.itos[i] for i in tokens)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, d_seq: int):
        super().__init__()
        positions = torch.arange(d_seq).unsqueeze(1)
        encodings = torch.zeros(d_seq, d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        encodings[:, 0::2] = torch.sin(positions * div_term)
        encodings[:, 1::2] = torch.cos(positions * div_term)
        self.encodings: Tensor  # required for the type checker
        self.register_buffer("encodings", encodings)

    def forward(self, input: Tensor) -> Tensor:
        return input + self.encodings
