import torch
from torch import Tensor
import torch.nn as nn

from ml_translate.attention import scaled_dot_product_attention


class EncoderDecoder(nn.Module):
    def __init__(self, d_model, seq_len, n_layers, ff_d_hidden):
        super().__init__()
        self.encoder = Encoder(
            d_model=d_model, seq_len=seq_len, n_layers=n_layers, ff_d_hidden=ff_d_hidden
        )
        self.decoder = Decoder(
            d_model=d_model, seq_len=seq_len, n_layers=n_layers, ff_d_hidden=ff_d_hidden
        )

    def encode(self, input: Tensor) -> Tensor:
        x = self.encoder.forward(input)
        return x

    def decode(self, input: Tensor, encoder_output: Tensor) -> Tensor:
        x = self.decoder.forward(input, encoder_output)
        return x

    def forward(self, src_input: Tensor, target_input) -> Tensor:
        encoder_output = self.encode(src_input)
        decoder_output = self.decode(target_input, encoder_output)
        return decoder_output


class Encoder(nn.Module):
    def __init__(self, d_model: int, seq_len: int, n_layers: int, ff_d_hidden: int):
        super().__init__()
        self.layers = [
            EncoderLayer(d_model=d_model, seq_len=seq_len, ff_d_hidden=ff_d_hidden)
            for _ in range(n_layers)
        ]

    def forward(self, input: Tensor) -> Tensor:
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, seq_len: int, ff_d_hidden: int):
        super().__init__()
        self.multiHeadSublayer = MultiHeadSublayer(d_model=d_model, seq_len=seq_len)
        self.feedForwardSublayer = FeedForwardSublayer(
            d_model=d_model, d_hidden=ff_d_hidden
        )

    def forward(self, input: Tensor) -> Tensor:
        x = input
        x = self.multiHeadSublayer.forward(x)
        x = self.feedForwardSublayer.forward(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, seq_len: int, n_layers: int, ff_d_hidden: int):
        super().__init__()
        self.layers = [
            DecoderLayer(d_model=d_model, seq_len=seq_len, ff_d_hidden=ff_d_hidden)
            for _ in range(n_layers)
        ]

    def forward(self, input: Tensor, encoder_output: Tensor) -> Tensor:
        x = input
        for layer in self.layers:
            x = layer.forward(x, encoder_output)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, seq_len: int, ff_d_hidden: int):
        super().__init__()

        self.maskedMultiHeadSublayer = MultiHeadSublayer(
            d_model=d_model, seq_len=seq_len, mask=True
        )
        self.multiHeadSublayer = MultiHeadSublayer(d_model=d_model, seq_len=seq_len)
        self.feedForwardSublayer = FeedForwardSublayer(
            d_model=d_model, d_hidden=ff_d_hidden
        )

    def forward(self, input: Tensor, encoder_output: Tensor) -> Tensor:
        x = input
        x = self.maskedMultiHeadSublayer.forward(x)
        x = self.multiHeadSublayer.forward(
            x, override_k=encoder_output, override_v=encoder_output
        )
        x = self.feedForwardSublayer.forward(x)
        return x


class MultiHeadSublayer(nn.Module):
    def __init__(self, d_model: int, seq_len: int, mask: bool = False):
        super().__init__()
        self.multiHeadAttention = MultiHeadAttention(
            n_heads=8, d_model=d_model, seq_len=seq_len, mask=mask
        )
        self.addAndNorm = AddAndNorm(d_model=d_model)

    def forward(
        self,
        input: Tensor,
        override_k: Tensor | None = None,
        override_v: Tensor | None = None,
    ) -> Tensor:
        k = override_k if override_k is not None else input
        v = override_v if override_v is not None else input
        q = input
        x = self.multiHeadAttention.forward(v, k, q)
        x = self.addAndNorm.forward(input, x)
        return x


class FeedForwardSublayer(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.feedForward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.addAndNorm = AddAndNorm(d_model=d_model)

    def forward(self, input: Tensor) -> Tensor:
        x = input
        x = self.feedForward.forward(x)
        x = self.addAndNorm.forward(input, x)
        return x


class AddAndNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.norm = Norm(d_model=d_model, eps=eps)

    def forward(self, input: Tensor, sublayer_output: Tensor) -> Tensor:
        # diverging from paper for better gradient flow
        # paper gives LayerNorm(x + Sublayer(x))
        return input + self.norm.forward(sublayer_output)


class Norm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        mean = torch.mean(input, dim=-1, keepdim=True)
        var = torch.var(input, dim=-1, keepdim=True)
        return self.gamma * (input - mean) / torch.sqrt(var + self.eps) + self.beta


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones((d_model, d_hidden)))
        self.b1 = nn.Parameter(torch.zeros(d_hidden))
        self.w2 = nn.Parameter(torch.ones((d_hidden, d_model)))
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, input: Tensor) -> Tensor:
        ReLU = nn.ReLU()
        return ReLU(input @ self.w1 + self.b1) @ self.w2 + self.b2


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, seq_len: int, mask: bool = False):
        super().__init__()
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.heads = [
            SingleHeadAttention(
                d_in=d_model, d_out=d_model // n_heads, seq_len=seq_len, mask=mask
            )
            for _ in range(n_heads)
        ]
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, v: Tensor, k: Tensor, q: Tensor) -> Tensor:
        outputs = [head.forward(v, k, q) for head in self.heads]
        x = torch.concat(outputs, dim=-1)
        return self.out_linear.forward(x)


class SingleHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, seq_len: int, mask: bool = False):
        super().__init__()
        self.v_linear = nn.Linear(in_features=d_in, out_features=d_out)
        self.k_linear = nn.Linear(in_features=d_in, out_features=d_out)
        self.q_linear = nn.Linear(in_features=d_in, out_features=d_out)
        self.mask = None
        if mask:
            ones = torch.ones((seq_len, seq_len))
            self.mask = torch.triu(ones, diagonal=1)

    def forward(self, v: Tensor, k: Tensor, q: Tensor) -> Tensor:
        l_v = self.v_linear.forward(v)
        l_k = self.k_linear.forward(k)
        l_q = self.q_linear.forward(q)
        return scaled_dot_product_attention(l_q, l_k, l_v, self.mask)
