class EncoderStack:
    def __init__(self, N=6):
        self.N = N
        self.stack = [EncoderLayer() for _ in range(N)]

    def forward(self, X):
        for layer in self.stack:
            X = layer.forward(X)
        return X


class EncoderLayer:
    def __init__(self):
        pass

    def forward(self, X):
        return X


class DecoderStack:
    def __init__(self, N=6):
        self.N = N

    def forward(self, X):
        return X


class Transformer:
    def __init__(self):
        self.encoder_stack = EncoderStack()
        self.decoder_stack = DecoderStack()

    def forward(self, X):
        return self.decoder_stack.forward(self.encoder_stack.forward(X))


"""
transformer = Transformer()
transformer.fit_transform(X)
transformer.predict(X)
"""
