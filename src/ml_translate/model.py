import numpy as np
import torch.nn as nn


def ReLU(x):
    return np.maximum(0, x)


def softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = np.size(k, axis=-1)
    inter = q @ k.T
    inter = inter / np.sqrt(d_k)
    if mask is not None:
        inter = np.where(mask == 0, -np.inf, inter)
    inter = softmax(inter)
    return v @ inter.T


class Norm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class Linear:
    def __init__(self, in_features, out_features):
        # TODO make these learnable
        self.w = np.ones((in_features, out_features))
        self.b = np.zeros(out_features)

    def forward(self, x):
        return x @ self.w + self.b


class FeedForward:
    def __init__(self, d_model, d_hidden=2048):
        # TODO make these learnable
        self.w1 = np.ones((d_model, d_hidden))
        self.b1 = np.zeros(d_hidden)
        self.w2 = np.ones((d_hidden, d_model))
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        return ReLU(x @ self.w1 + self.b1) @ self.w2 + self.b2


class AddAndNorm:
    def __init__(self, shape, eps=1e-6):
        # TODO make these learnable
        self.weights = np.ones(shape)
        self.biases = np.zeros(shape)
        self.norm = Norm(shape, eps)

    def forward(self, x, sublayer_output):
        return self.weights * self.norm.forward(x + sublayer_output) + self.biases


class SingleHeadAttention:
    def __init__(self, d_in, d_out, mask=None):
        self.v_linear = Linear(d_in, d_out)
        self.k_linear = Linear(d_in, d_out)
        self.q_linear = Linear(d_in, d_out)
        self.mask = mask

    def forward(self, v, k, q):
        l_v = self.v_linear.forward(v)
        l_k = self.k_linear.forward(k)
        l_q = self.q_linear.forward(q)
        return scaled_dot_product_attention(l_q, l_k, l_v, self.mask)


class MultiHeadAttention:
    # h is number of heads
    def __init__(self, h, d_model, mask=None):
        self.h = h
        self.d_model = d_model
        self.heads = [SingleHeadAttention(d_model, d_model / h, mask) for _ in range(h)]
        self.out_linear = Linear(d_model, d_model)

    def forward(self, v, k, q):
        outputs = [head.forward(v, k, q) for head in self.heads]
        x = np.concat(outputs)
        return self.out_linear.forward(x)


class Encoder:
    def __init__(self, size, n_layers, ff_d_hidden):
        self.layers = [EncoderLayer(size, ff_d_hidden) for _ in range(n_layers)]

    def forward(self, input):
        # some embedding encoding
        # positional encoding
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x


class EncoderLayer:
    def __init__(self, size, d_hidden):
        self.multiHeadSublayer = MultiHeadSublayer(size)
        self.feedForwardSublayer = FeedForwardSublayer(size, d_hidden)

    def forward(self, input):
        x = input
        x = self.multiHeadSublayer.forward(x)
        x = self.feedForwardSublayer.forward(x)
        return x


class Decoder:
    def __init__(self, size, n_layers, ff_d_hidden):
        self.layers = [DecoderLayer(size, ff_d_hidden) for _ in range(n_layers)]

    def forward(self, input, encoder_output):
        # some embedding encoding
        # positional embedding
        x = input
        for layer in self.layers:
            x = layer.forward(x, encoder_output)
        return x


class DecoderLayer:
    def __init__(self, size, d_hidden):
        self.maskedMultiHeadSublayer = MultiHeadSublayer(size, mask=None)
        self.multiHeadSublayer = MultiHeadSublayer(size)
        self.feedForwardSublayer = FeedForwardSublayer(size, d_hidden)

    def forward(self, input, encoder_output):
        # need to do the right shift
        x = input
        x = self.maskedMultiHeadSublayer.forward(x)
        x = self.multiHeadSublayer.forward(
            x, override_k=encoder_output, override_v=encoder_output
        )
        x = self.feedForwardSublayer


class MultiHeadSublayer:
    def __init__(self, size, mask=None):
        self.multiHeadAttention = MultiHeadAttention(size, 8, mask)
        self.addAndNorm = AddAndNorm(size)

    def forward(self, input, override_k=None, override_v=None):
        k = override_k or input.clone()
        v = override_v or input.clone()
        q = input.clone()
        x = self.multiHeadAttention.forward(v, k, q)
        x = self.addAndNorm.forward(input, x)
        return x


class FeedForwardSublayer:
    def __init__(self, size, d_hidden):
        self.feedForward = FeedForward(size, d_hidden)
        self.addAndNorm = AddAndNorm(size)

    def forward(self, input):
        x = input
        x = self.feedForward.forward(x)
        x = self.addAndNorm.forward(input, x)
        return x


class Transformer:
    # size is 2-dim, first is sequence len, second is embedding size 512
    def __init__(self, size, layers, ff_d_hidden):
        self.size = size
        self.layers = layers
        self.encoder = Encoder(size, layers, ff_d_hidden)
        self.decoder = Decoder(size, layers, ff_d_hidden)

    def encode(self, input):
        self.encoder.forward(input)
