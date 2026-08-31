import torch
from torch import Tensor, nn


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal_mask: Tensor | None = None,
    pad_mask: Tensor | None = None,
) -> Tensor:
    d_k = k.size(dim=-1)

    inter = q @ k.mT
    inter = inter / d_k**0.5

    if causal_mask is not None:
        inter = torch.where(causal_mask, -torch.inf, inter)
    if pad_mask is not None:
        reshaped = pad_mask.unsqueeze(-2)
        inter = torch.where(reshaped, -torch.inf, inter)

    softmax = nn.Softmax(dim=-1)
    inter = softmax(inter)
    return inter @ v
