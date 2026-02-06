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
    """Additive attention mechanism (Bahdanau et al., 2015).

    Computes attention scores using a learned alignment model:
    score = V * tanh(W * query + U * keys)
    """

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


class LuongAttention(nn.Module):
    """Multiplicative attention mechanism (Luong et al., 2015).

    Supports three scoring methods:
    - "dot": score = query · keys
    - "general": score = query · W · keys
    - "concat": score = V · tanh(W · [query; keys])
    """

    def __init__(self, hidden_size: int, method: str = "general"):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size

        if method == "general":
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            self.W = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.V = nn.Linear(hidden_size, 1, bias=False)
        elif method != "dot":
            raise ValueError(f"Unknown attention method: {method}. Use 'dot', 'general', or 'concat'.")

    def forward(self, query: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            query: Decoder hidden state (batch, 1, hidden_size)
            keys: Encoder outputs (batch, seq_len, hidden_size)

        Returns:
            context: Weighted sum of keys (batch, 1, hidden_size)
            weights: Attention weights (batch, 1, seq_len)
        """
        if self.method == "dot":
            # score = query · keys^T
            scores = torch.bmm(query, keys.transpose(1, 2))
        elif self.method == "general":
            # score = query · W · keys^T
            scores = torch.bmm(self.W(query), keys.transpose(1, 2))
        else:  # concat
            # Expand query to match keys sequence length
            seq_len = keys.size(1)
            query_expanded = query.expand(-1, seq_len, -1)
            # Concatenate and compute score
            concat = torch.cat([query_expanded, keys], dim=2)
            scores = self.V(torch.tanh(self.W(concat)))
            scores = scores.transpose(1, 2)

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
        attention_type: str = "bahdanau",
    ):
        """
        Args:
            hidden_size: Size of hidden states.
            output_size: Size of output vocabulary.
            dropout_p: Dropout probability.
            device: Device to use for tensors.
            embedding: Optional pre-trained embedding layer.
            attention_type: Type of attention mechanism. Options:
                - "bahdanau": Additive attention (default)
                - "luong_dot": Luong dot-product attention
                - "luong_general": Luong general attention
                - "luong_concat": Luong concat attention
        """
        super().__init__()
        self.device = device
        self.attention_type = attention_type

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(output_size, hidden_size)

        # Create attention mechanism
        if attention_type == "bahdanau":
            self.attention = BahdanauAttention(hidden_size)
        elif attention_type.startswith("luong"):
            method = attention_type.split("_")[1] if "_" in attention_type else "general"
            self.attention = LuongAttention(hidden_size, method=method)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

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
