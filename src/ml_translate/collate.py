from tokenizers import Tokenizer
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ml_translate.config import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


def collate(
    batch: tuple[str, ...], tokenizer: Tokenizer
) -> tuple[Tensor, Tensor, Tensor]:
    BOS_ID: int | None = tokenizer.token_to_id(BOS_TOKEN)
    EOS_ID: int | None = tokenizer.token_to_id(EOS_TOKEN)
    PAD_ID: int | None = tokenizer.token_to_id(PAD_TOKEN)
    assert BOS_ID is not None
    assert EOS_ID is not None
    assert PAD_ID is not None
    tokens: list[Tensor] = [
        torch.tensor([BOS_ID, *tokenizer.encode(item).ids, EOS_ID], dtype=torch.long)
        for item in batch
    ]
    padded: Tensor = pad_sequence(tokens, batch_first=True, padding_value=float(PAD_ID))
    pad_mask: Tensor = torch.where(padded == float(PAD_ID), 0.0, 1.0)

    ones = torch.ones(padded.shape)
    causal_mask = torch.triu(ones, diagonal=1)
    return padded, pad_mask, causal_mask
