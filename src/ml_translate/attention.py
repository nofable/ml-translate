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
    inter = inter / torch.sqrt(torch.tensor([d_k]))

    if causal_mask is not None:
        inter = torch.where(causal_mask == 0, -torch.inf, inter)
    if pad_mask is not None:
        reshaped = pad_mask.unsqueeze(-2)
        inter = torch.where(reshaped == 0, -torch.inf, inter)

    softmax = nn.Softmax(dim=-1)
    inter = softmax(inter)
    return inter @ v
