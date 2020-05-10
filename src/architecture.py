import random
from .Tensor import Tensor

class Neuron:
    def __init__(self, in_c: int, name=''):
        self.name = name
        self.w = [Tensor(random.normalvariate(0, 0.1),
                         name='{}+p{}'.format(name, n))
                  for n in range(in_c)]
        self.b = Tensor(random.normalvariate(0, 0.1),
                        name='{}_bias'.format(name))

    def __call__(self, x: list):
        out = Tensor(0)

        for n in range(len(x)):
            out = out + self.w[n] * x[n]
        out = out + self.b

        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, in_c, out_c, name=''):
        self.name = name
        self.neurons = [Neuron(in_c, name='{}n{}'.format(name, n))
                        for n in range(out_c)]
        
    def __call__(self, x: list):
        return [n(x) for n in self.neurons]

    def parameters(self):
        parameterList = []

        for n in self.neurons:
            parameterList += n.parameters()

        return parameterList

class Network:
    def __init__(self, layers: list, name=''):
        self.layers = layers
        self.name = name

    def __call__(self, x):
        for l in self.layers:
            x = l(x)

        return x

    def parameters():
        parameterList = []

        for l in self.layers:
            parameterList += l.parameters()

        return parameterList

    def __repr__(self):
        return self.name
        #TODO

class MLP:
    def __init__(self, in_c, hidden):
        layers = []
        
        last_channels = in_c
        for n, h in enumerate(hidden):
            layers.append(Layer(last_channels, h, name='l{}'.format(n)))
            last_channels = h

        self.network = Network(layers, name='MLP')

    def __call__(self, x):
        return self.network(x)

    def parameters():
        return self.network.parameters()

    def __repr__(self):
        return self.network.__repr__()