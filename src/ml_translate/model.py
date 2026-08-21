import numpy as np


def ReLU(x):
    return np.maximum(0, x)


def softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


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


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = np.size(k, axis=-1)
    inter = (q @ k.T) / np.sqrt(d_k)
    if mask is not None:
        inter = np.where(mask == 0, -np.inf, inter)
    inter = softmax(inter)
    return inter @ v
