import numpy as np


# class EncoderStack:
#     def __init__(self, N=6):
#         self.N = N
#         self.stack = [EncoderLayer() for _ in range(N)]

#     def forward(self, X):
#         for layer in self.stack:
#             X = layer.forward(X)
#         return X


# class EncoderLayer:
#     def __init__(self):
#         self.sublayer1 = SubLayerWithAddAndNorm(MultiHeadAttention())
#         self.sublayer2 = SubLayerWithAddAndNorm(FeedForward())

#     def forward(self, X):
#         X = self.sublayer1.forward(X)
#         X = self.sublayer2.forward(X)
#         return X


# class SubLayerWithAddAndNorm:
#     def __init__(self, sublayer):
#         self.sublayer = sublayer
#         self.add_and_norm = AddAndNorm()

#     def forward(self, X):
#         sublayer_output = self.sublayer.forward(X.copy())
#         X = self.add_and_norm.forward(X, sublayer_output)
#         return X


class AddAndNorm:
    def __init__(self, shape, eps=1e-6):
        self.eps = eps
        # TODO make these learnable
        self.weights = np.ones(shape)
        self.biases = np.zeros(shape)

    def layer_norm(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps)

    def forward(self, x, sublayer_output):
        return (self.weights * self.layer_norm(x + sublayer_output)) + self.biases


# class MultiHeadAttention:
#     def __init__(self):
#         pass

#     def forward(self, X):
#         return X


class FeedForward:
    """
    "...two linear transformations with a ReLU activation in between."
    512 -> 2048 -> 512
    """

    def __init__(self, d_model, d_hidden=2048):
        # TODO make these learnable
        self.w1 = np.ones((d_model, d_hidden))
        self.b1 = np.zeros(d_hidden)
        self.w2 = np.ones((d_hidden, d_model))
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        print(x @ self.w1 + self.b1)
        return ReLU(x @ self.w1 + self.b1) @ self.w2 + self.b2


def ReLU(x):
    return np.maximum(0, x)


# class DecoderStack:
#     def __init__(self, N=6):
#         self.N = N

#     def forward(self, X):
#         return X


# class Transformer:
#     def __init__(self):
#         self.encoder_stack = EncoderStack()
#         self.decoder_stack = DecoderStack()

#     def forward(self, X):
#         return self.decoder_stack.forward(self.encoder_stack.forward(X))
