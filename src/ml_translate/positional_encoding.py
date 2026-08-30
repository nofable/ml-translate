import torch
from torch import nn, Tensor


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, p_dropout: float = 0):
        super().__init__()
        self.p_dropout = p_dropout
        positions = torch.arange(max_seq_len).unsqueeze(1)
        encodings = torch.zeros(max_seq_len, d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        encodings[:, 0::2] = torch.sin(positions * div_term)
        encodings[:, 1::2] = torch.cos(positions * div_term)
        self.encodings: Tensor  # required for the type checker
        self.register_buffer("encodings", encodings)

    def forward(self, input: Tensor) -> Tensor:
        dropout = nn.Dropout(p=self.p_dropout)
        return dropout(input + self.encodings[: input.size(1)])
