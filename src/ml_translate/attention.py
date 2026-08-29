import torch
from torch import nn, Tensor


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
) -> Tensor:
    d_k = k.size(dim=-1)
    inter = q @ k.T
    inter = inter / torch.sqrt(torch.tensor([d_k]))
    if mask is not None:
        inter = torch.where(mask == 0, -torch.inf, inter)
    softmax = nn.Softmax(dim=-1)
    inter = softmax(inter)
    return inter @ v
