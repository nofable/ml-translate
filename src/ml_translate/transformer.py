import torch
from torch import Tensor
import torch.nn as nn

from ml_translate.encoder_decoder import EncoderDecoder


class Transformer(nn.Module):
    def __init__(self, d_model: int, d_seq: int, n_vocab: int):
        super().__init__()
        self.positionalEncoder = PositionalEncoder(d_model, d_seq)
        self.encoderDecoder = EncoderDecoder(
            d_model=d_model, n_layers=6, ff_d_hidden=1000
        )
        self.encoder_embed = nn.Embedding(num_embeddings=n_vocab, embedding_dim=d_model)
        self.decoder_embed = nn.Embedding(num_embeddings=n_vocab, embedding_dim=d_model)

    def forward(self, encoder_input, decoder_input):
        embedded_encoder_input = self.encoder_embed(encoder_input)
        pe_encoder_input = self.positionalEncoder.forward(embedded_encoder_input)

        embedded_decoder_input = self.decoder_embed(decoder_input)
        pe_decoder_input = self.positionalEncoder(embedded_decoder_input)

        return self.encoderDecoder.forward(pe_encoder_input, pe_decoder_input)


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
