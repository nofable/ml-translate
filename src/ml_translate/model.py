import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ml_translate.config import default_config
from ml_translate.data import SOS_token, MAX_LENGTH


class EncoderRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_p: float = default_config.dropout_p,
        embedding: nn.Module | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        device: torch.device | None = None,
        embedding: nn.Module | None = None,
    ):
        super().__init__()
        self.device = device
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        encoder_outputs: Tensor,
        encoder_hidden: Tensor,
        target_tensor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, None]:
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=self.device
        ).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        # Use target length during training, MAX_LENGTH during inference
        max_len = target_tensor.size(1) if target_tensor is not None else MAX_LENGTH

        for i in range(max_len):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input

        decoder_outputs_cat = torch.cat(decoder_outputs, dim=1)
        decoder_outputs_cat = F.log_softmax(decoder_outputs_cat, dim=-1)
        return (
            decoder_outputs_cat,
            decoder_hidden,
            None,
        )  # We return `None` for consistency in the training loop

    def forward_step(self, input: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        dropout_p: float = default_config.dropout_p,
        device: torch.device | None = None,
        embedding: nn.Module | None = None,
    ):
        super().__init__()
        self.device = device
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        encoder_outputs: Tensor,
        encoder_hidden: Tensor,
        target_tensor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=self.device
        ).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs: list[Tensor] = []
        attentions: list[Tensor] = []

        # Use target length during training, MAX_LENGTH during inference
        max_len = target_tensor.size(1) if target_tensor is not None else MAX_LENGTH

        for i in range(max_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input

        decoder_outputs_cat = torch.cat(decoder_outputs, dim=1)
        decoder_outputs_cat = F.log_softmax(decoder_outputs_cat, dim=-1)
        attentions_cat = torch.cat(attentions, dim=1)

        return decoder_outputs_cat, decoder_hidden, attentions_cat

    def forward_step(
        self, inpt: Tensor, hidden: Tensor, encoder_outputs: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        embedded = self.dropout(self.embedding(inpt))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
