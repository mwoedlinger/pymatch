from .Neuron import Neuron

class Layer:
    """
    A layer is a list of neurons.
    """
    def __init__(self, in_c: int, out_c: int, activationFct, name=''):
        self.f = activationFct
        self.name = name
        self.neurons = [Neuron(in_c, name+'_n'+str(n)) for n in range(out_c)]
        self.forward = []

    def __call__(self, x: list):
        self.forward = [self.f(n(x)) for n in self.neurons]
        return self.forward

    def zero_grad(self):
        for n in self.neurons:
            n.zero_grad()

    def backward(self):
        for n in self.forward:
            n.grad = n.grad*self.f.backward_call(n.val)
            n.backward()

    def optimize(self, lr):
        for n in self.neurons:
            n.optimize(lr)

    def parameters(self):
        parameters = []

        for n in self.neurons:
            parameters += n.parameters()

        return parameters 