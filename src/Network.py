class Network:
    """
    A Network is a list of layers.
    """
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for l in self.layers:
            x = l(x)

        return x

    def zero_grad(self):
        for l in reversed(self.layers):
            l.zero_grad()

    def backward(self):
        for l in reversed(self.layers):
            l.backward()

    def optimize(self, lr):
        for l in reversed(self.layers):
            l.optimize(lr)

    def parameters(self):
        parameters = []

        for l in self.layers:
            parameters += l.parameters()

        return parameters 