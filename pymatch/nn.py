import random
from .variable import Variable
from .functions import relu, softmax, sigmoid

class Neuron:
    """
    A Neuron computes a linear combination of the feature vector where weights are
    given by 'Variable' objects to keep track of the computation graph.
    """
    def __init__(self, in_c: int, name='', bias=True):
        self.name = name
        self.bias = bias

        self.w = [Variable(random.uniform(-1, 1),
                         name='{}+p{}'.format(name, n))
                  for n in range(in_c)]
        
        if self.bias:
            self.b = Variable(random.normalvariate(0, 0.1),
                            name='{}_bias'.format(name))

    def __call__(self, x: list):
        out = Variable(0)

        for n in range(len(x)):
            out = out + self.w[n] * x[n]
        if self.bias:
            out = out + self.b

        return out

    def parameters(self):
        if self.bias:
            return self.w + [self.b]
        else:
            return self.w


class Layer:
    """
    A layer is a collection of neurons.
    """
    def __init__(self, in_c, out_c, name='', bias=True, activation='relu'):
        self.in_c = in_c
        self.out_c = out_c
        self.name = name

        self.activation = activation
        self.neurons = [Neuron(in_c, name='{}n{}'.format(name, n), bias=bias)
                        for n in range(out_c)]
        
    def __call__(self, x: list):
        if self.activation == 'relu':
            return [relu(n(x)) for n in self.neurons]
        elif self.activation == 'identity':
            return [n(x) for n in self.neurons]
        elif self.activation == 'softmax':
            return softmax([n(x) for n in self.neurons])
        elif self.activation == 'sigmoid':
            return [sigmoid(n(x)) for n in self.neurons]

    def parameters(self):
        parameterList = []

        for n in self.neurons:
            parameterList += n.parameters()

        return parameterList

    def __repr__(self):
        return '{:10} (in_c = {}, out_c = {}, act = {}'.format('fcn', self.in_c, self.out_c, self.activation)

class Network:
    """
    A network is a collection of layers.
    """
    def __init__(self, layers: list, name=''):
        self.layers = layers
        self.name = name

    def __call__(self, x):
        for l in self.layers:
            x = l(x)

        return x

    def parameters(self):
        parameterList = []

        for l in self.layers:
            parameterList += l.parameters()

        return parameterList

    def __repr__(self):
        repr_string = self.name + ':\n'
        for l in self.layers:
            repr_string += str(l) + '\n'

        return repr_string

class MLP:
    """
    A multilayer perceptron.
    """
    def __init__(self, in_c, hidden, activation='identity'):
        layers = []
        
        last_channels = in_c
        for n in range(len(hidden)-1):
            h = hidden[n]

            layers.append(Layer(last_channels, h, name='l{}'.format(n), activation='relu'))
            last_channels = h
        
        if hidden[-1] == 1:
            activation = activation

        layers.append(Layer(last_channels, hidden[-1], name='l{}'.format(n), activation=activation))

        self.network = Network(layers, name='MLP')

    def __call__(self, x):
        return self.network(x)

    def parameters(self):
        return self.network.parameters()

    def __repr__(self):
        return self.network.__repr__()