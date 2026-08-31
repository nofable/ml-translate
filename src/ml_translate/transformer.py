import math

import torch
import torch.nn as nn
from torch import Tensor

from ml_translate.attention import scaled_dot_product_attention
from ml_translate.positional_encoding import PositionalEncoder


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_embeddings: int,
        max_seq_len: int,
        n_layers: int,
        ff_d_hidden: int,
        p_dropout: float,
    ):
        super().__init__()

        self.d_model = d_model
        self.positionalEncoder = PositionalEncoder(
            d_model=d_model, max_seq_len=max_seq_len
        )
        self.encoderDecoder = EncoderDecoder(
            d_model=d_model,
            n_layers=n_layers,
            ff_d_hidden=ff_d_hidden,
            p_dropout=p_dropout,
            max_seq_len=max_seq_len,
        )
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=d_model
        )
        # crucial for bias to be false in order to share weights with embedding
        self.output_linear = nn.Linear(
            in_features=d_model, out_features=num_embeddings, bias=False
        )

        # share weight matrix between the two enbedding layers and the pre-softmanx linear
        self.output_linear.weight = self.embedding.weight

    def forward(self, inputs, outputs, inputs_pad_mask, outputs_pad_mask):
        # In the embedding layers we multiply those weights by sqrt of d_model
        x_inputs = self.embedding(inputs) * math.sqrt(self.d_model)
        x_inputs = self.positionalEncoder.forward(x_inputs)

        # In the embedding layers we multiply those weights by sqrt of d_model
        x_outputs = self.embedding(outputs) * math.sqrt(self.d_model)
        x_outputs = self.positionalEncoder.forward(x_outputs)

        decoded = self.encoderDecoder.forward(
            x_inputs,
            x_outputs,
            inputs_pad_mask=inputs_pad_mask,
            outputs_pad_mask=outputs_pad_mask,
        )
        return self.output_linear.forward(decoded)


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        ff_d_hidden: int,
        p_dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.encoder = Encoder(
            d_model=d_model,
            n_layers=n_layers,
            ff_d_hidden=ff_d_hidden,
            p_dropout=p_dropout,
            max_seq_len=max_seq_len,
        )
        self.decoder = Decoder(
            d_model=d_model,
            n_layers=n_layers,
            ff_d_hidden=ff_d_hidden,
            p_dropout=p_dropout,
            max_seq_len=max_seq_len,
        )

    def encode(self, inputs: Tensor, inputs_pad_mask: Tensor) -> Tensor:
        x = self.encoder.forward(inputs=inputs, inputs_pad_mask=inputs_pad_mask)
        return x

    def decode(
        self,
        outputs: Tensor,
        encoder_output: Tensor,
        outputs_pad_mask: Tensor,
        inputs_pad_mask: Tensor,
    ) -> Tensor:
        x = self.decoder.forward(
            outputs=outputs,
            encoder_output=encoder_output,
            inputs_pad_mask=inputs_pad_mask,
            outputs_pad_mask=outputs_pad_mask,
        )
        return x

    def forward(
        self,
        inputs: Tensor,
        outputs: Tensor,
        inputs_pad_mask: Tensor,
        outputs_pad_mask: Tensor,
    ) -> Tensor:
        encoder_output = self.encode(inputs=inputs, inputs_pad_mask=inputs_pad_mask)
        decoder_output = self.decode(
            outputs=outputs,
            encoder_output=encoder_output,
            outputs_pad_mask=outputs_pad_mask,
            inputs_pad_mask=inputs_pad_mask,
        )
        return decoder_output


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        ff_d_hidden: int,
        p_dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.layers = [
            EncoderLayer(
                d_model=d_model,
                ff_d_hidden=ff_d_hidden,
                p_dropout=p_dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ]

    def forward(self, inputs: Tensor, inputs_pad_mask: Tensor) -> Tensor:
        x = inputs
        for layer in self.layers:
            x = layer.forward(inputs=x, inputs_pad_mask=inputs_pad_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, ff_d_hidden: int, p_dropout: float, max_seq_len: int
    ):
        super().__init__()
        self.multiHeadSublayer = MultiHeadSublayer(
            d_model=d_model, p_dropout=p_dropout, max_seq_len=max_seq_len
        )
        self.feedForwardSublayer = FeedForwardSublayer(
            d_model=d_model, d_hidden=ff_d_hidden, p_dropout=p_dropout
        )

    def forward(self, inputs: Tensor, inputs_pad_mask: Tensor) -> Tensor:
        x = inputs
        x = self.multiHeadSublayer.forward(inputs=x, pad_mask=inputs_pad_mask)
        x = self.feedForwardSublayer.forward(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        ff_d_hidden: int,
        p_dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.layers = [
            DecoderLayer(
                d_model=d_model,
                ff_d_hidden=ff_d_hidden,
                p_dropout=p_dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ]

    def forward(
        self,
        outputs: Tensor,
        encoder_output: Tensor,
        outputs_pad_mask: Tensor,
        inputs_pad_mask: Tensor,
    ) -> Tensor:
        x = outputs
        for layer in self.layers:
            x = layer.forward(
                outputs=x,
                encoder_output=encoder_output,
                outputs_pad_mask=outputs_pad_mask,
                inputs_pad_mask=inputs_pad_mask,
            )
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, ff_d_hidden: int, p_dropout: float, max_seq_len: int
    ):
        super().__init__()

        self.maskedMultiHeadSublayer = MultiHeadSublayer(
            d_model=d_model, p_dropout=p_dropout, max_seq_len=max_seq_len
        )
        self.multiHeadSublayer = MultiHeadSublayer(
            d_model=d_model, p_dropout=p_dropout, max_seq_len=max_seq_len
        )
        self.feedForwardSublayer = FeedForwardSublayer(
            d_model=d_model, d_hidden=ff_d_hidden, p_dropout=p_dropout
        )

    def forward(
        self,
        outputs: Tensor,
        encoder_output: Tensor,
        outputs_pad_mask: Tensor,
        inputs_pad_mask: Tensor,
    ) -> Tensor:
        x = outputs
        x = self.maskedMultiHeadSublayer.forward(inputs=x, pad_mask=outputs_pad_mask)
        x = self.multiHeadSublayer.forward(
            inputs=x,
            override_k=encoder_output,
            override_v=encoder_output,
            pad_mask=inputs_pad_mask,
        )
        x = self.feedForwardSublayer.forward(x)
        return x


class MultiHeadSublayer(nn.Module):
    def __init__(self, d_model: int, p_dropout: float, max_seq_len: int):
        super().__init__()
        self.p_dropout = p_dropout
        self.multiHeadAttention = MultiHeadAttention(
            n_heads=8, d_model=d_model, max_seq_len=max_seq_len
        )
        self.addAndNorm = AddAndNorm(d_model=d_model)

    def forward(
        self,
        inputs: Tensor,
        pad_mask: Tensor,
        override_k: Tensor | None = None,
        override_v: Tensor | None = None,
    ) -> Tensor:
        k = override_k if override_k is not None else inputs
        v = override_v if override_v is not None else inputs
        q = inputs

        x = self.multiHeadAttention.forward(v, k, q, pad_mask=pad_mask)
        dropout = nn.Dropout(p=self.p_dropout)
        x = dropout(x)
        x = self.addAndNorm.forward(inputs, x)
        return x


class FeedForwardSublayer(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, p_dropout: float):
        super().__init__()
        self.p_dropout = p_dropout
        self.feedForward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.addAndNorm = AddAndNorm(d_model=d_model)

    def forward(self, input: Tensor) -> Tensor:
        x = input
        x = self.feedForward.forward(x)
        dropout = nn.Dropout(p=self.p_dropout)
        x = dropout(x)
        x = self.addAndNorm.forward(input, x)
        return x


class AddAndNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = Norm(d_model=d_model)

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
    def __init__(self, n_heads: int, d_model: int, max_seq_len: int):
        super().__init__()
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.heads = [
            SingleHeadAttention(
                d_in=d_model, d_out=d_model // n_heads, max_seq_len=max_seq_len
            )
            for _ in range(n_heads)
        ]
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(
        self,
        v: Tensor,
        k: Tensor,
        q: Tensor,
        pad_mask: Tensor,
    ) -> Tensor:
        outputs = [
            head.forward(v=v, k=k, q=q, pad_mask=pad_mask) for head in self.heads
        ]
        x = torch.concat(outputs, dim=-1)
        return self.out_linear.forward(x)


class SingleHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, max_seq_len: int):
        super().__init__()
        self.v_linear = nn.Linear(in_features=d_in, out_features=d_out)
        self.k_linear = nn.Linear(in_features=d_in, out_features=d_out)
        self.q_linear = nn.Linear(in_features=d_in, out_features=d_out)
        self.causal_mask: Tensor
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
            ),
        )

    def forward(
        self,
        v: Tensor,
        k: Tensor,
        q: Tensor,
        pad_mask: Tensor,
    ) -> Tensor:
        l_v = self.v_linear.forward(v)
        l_k = self.k_linear.forward(k)
        l_q = self.q_linear.forward(q)
        return scaled_dot_product_attention(
            q=l_q,
            k=l_k,
            v=l_v,
            causal_mask=self.causal_mask[: l_q.size(-2), : l_k.size(-2)],
            pad_mask=pad_mask,
        )
