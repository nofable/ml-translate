import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = k.size(dim=-1)  # size of the token vector (inner most)
    inter = q @ k.T
    inter = inter / torch.sqrt(torch.tensor([d_k]))
    if mask is not None:
        inter = torch.where(mask == 0, -torch.inf, inter)
    softmax = nn.Softmax(dim=-1)
    inter = softmax(inter)
    return inter @ v


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        # parameters shared across positions
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, input):
        mean = torch.mean(input, dim=-1, keepdim=True)
        var = torch.var(input, dim=-1, keepdim=True)
        return self.gamma * (input - mean) / torch.sqrt(var + self.eps) + self.beta


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden=2048):
        super().__init__()
        # parameters shared across positions
        self.w1 = nn.Parameter(torch.ones((d_model, d_hidden)))
        self.b1 = nn.Parameter(torch.zeros(d_hidden))
        self.w2 = nn.Parameter(torch.ones((d_hidden, d_model)))
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, input):
        ReLU = nn.ReLU()
        return ReLU(input @ self.w1 + self.b1) @ self.w2 + self.b2


class AddAndNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.norm = Norm(d_model=d_model, eps=eps)

    def forward(self, input, sublayer_output):
        # diverging from paper for better gradient flow
        # paper gives LayerNorm(x + Sublayer(x))
        return input + self.norm.forward(sublayer_output)


class SingleHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, mask=None):
        super().__init__()
        self.v_linear = nn.Linear(in_features=d_in, out_features=d_out)
        self.k_linear = nn.Linear(in_features=d_in, out_features=d_out)
        self.q_linear = nn.Linear(in_features=d_in, out_features=d_out)
        self.mask = mask

    def forward(self, v, k, q):
        l_v = self.v_linear.forward(v)
        l_k = self.k_linear.forward(k)
        l_q = self.q_linear.forward(q)
        return scaled_dot_product_attention(l_q, l_k, l_v, self.mask)


class MultiHeadAttention(nn.Module):
    # h is number of heads
    def __init__(self, n_heads, d_model, mask=None):
        super().__init__()
        self.n_heads = n_heads
        # ensure this is a neat split
        assert d_model % n_heads == 0
        self.heads = [
            SingleHeadAttention(d_in=d_model, d_out=d_model / n_heads, mask=mask)
            for _ in range(n_heads)
        ]
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, v, k, q):
        outputs = [head.forward(v, k, q) for head in self.heads]
        x = torch.concat(outputs)
        return self.out_linear.forward(x)


class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, ff_d_hidden):
        super().__init__()
        self.layers = [
            EncoderLayer(d_model=d_model, ff_d_hidden=ff_d_hidden)
            for _ in range(n_layers)
        ]

    def forward(self, input):
        # some embedding encoding
        # positional encoding
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ff_d_hidden):
        super().__init__()
        self.multiHeadSublayer = MultiHeadSublayer(d_model=d_model)
        self.feedForwardSublayer = FeedForwardSublayer(
            d_model=d_model, d_hidden=ff_d_hidden
        )

    def forward(self, input):
        x = input
        x = self.multiHeadSublayer.forward(x)
        x = self.feedForwardSublayer.forward(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, ff_d_hidden):
        super().__init__()
        self.layers = [
            DecoderLayer(d_model=d_model, ff_d_hidden=ff_d_hidden)
            for _ in range(n_layers)
        ]

    def forward(self, input, encoder_output):
        # some embedding encoding
        # positional embedding
        x = input
        for layer in self.layers:
            x = layer.forward(x, encoder_output)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ff_d_hidden):
        super().__init__()
        self.maskedMultiHeadSublayer = MultiHeadSublayer(d_model=d_model, mask=None)
        self.multiHeadSublayer = MultiHeadSublayer(d_model=d_model)
        self.feedForwardSublayer = FeedForwardSublayer(
            d_model=d_model, d_hidden=ff_d_hidden
        )

    def forward(self, input, encoder_output):
        # need to do the right shift
        x = input
        x = self.maskedMultiHeadSublayer.forward(x)
        x = self.multiHeadSublayer.forward(
            x, override_k=encoder_output, override_v=encoder_output
        )
        x = self.feedForwardSublayer


class MultiHeadSublayer(nn.Module):
    def __init__(self, d_model, mask=None):
        super().__init__()
        self.multiHeadAttention = MultiHeadAttention(
            n_heads=8, d_model=d_model, mask=mask
        )
        self.addAndNorm = AddAndNorm(d_model=d_model)

    def forward(self, input, override_k=None, override_v=None):
        k = override_k or input.clone()
        v = override_v or input.clone()
        q = input.clone()
        x = self.multiHeadAttention.forward(v, k, q)
        x = self.addAndNorm.forward(input, x)
        return x


class FeedForwardSublayer(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.feedForward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.addAndNorm = AddAndNorm(d_model=d_model)

    def forward(self, input):
        x = input
        x = self.feedForward.forward(x)
        x = self.addAndNorm.forward(input, x)
        return x


class Transformer(nn.Module):
    # size is 2-dim, first is sequence len, second is embedding size 512
    def __init__(self, d_model, n_layers, ff_d_hidden):
        super().__init__()
        self.encoder = Encoder(
            d_model=d_model, n_layers=n_layers, ff_d_hidden=ff_d_hidden
        )
        self.decoder = Decoder(
            d_model=d_model, n_layers=n_layers, ff_d_hidden=ff_d_hidden
        )

    def encode(self, input):
        self.encoder.forward(input)

    def decode(self, input, encoder_output):
        self.decoder.forward(input, encoder_output)

    def transform(self, input):
        encoder_output = self.encode(input)
        decoder_output = self.decode(input, encoder_output)
        # need to do final linear projection to embedding dictionary size and softmax
        return decoder_output
