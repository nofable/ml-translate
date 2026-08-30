import math

import torch
import torch.nn as nn

from ml_translate.encoder_decoder import EncoderDecoder
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
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, inputs, outputs, inputs_pad_mask, outputs_pad_mask, outputs_causal_mask
    ):
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
            outputs_causal_mask=outputs_causal_mask,
        )
        posits = self.softmax(self.output_linear.forward(decoded))
        vocab_index = torch.argmax(posits, dim=-1)
        return vocab_index
