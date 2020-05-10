import random
from .Var import Var

class Neuron:
    """
    A fully connected neuron
    """
    def __init__(self, in_c: int, name=''):
        self.name = name
        self.vars = [Var(random.uniform(-0.2, 0.2), name+'_v'+str(n)) for n in range(in_c)]
        self.var_b = Var(random.uniform(-0.1, 0.1), name+'_bias')

    def __call__(self, x: list):
        """
        Apply neuron to data. Assumes that len(x) == len(self.vars)
        """
        forward = Var(0, self.name+'_forward')

        for n, inp in enumerate(x):
            forward = forward + self.vars[n]*inp

        self.forward = forward + self.var_b

        return self.forward

    def zero_grad(self):
        for v in self.vars:
            v.zero_grad()
        self.var_b.zero_grad()

    def backward(self):
        self.forward.backward()

    def optimize(self, lr):
        for v in self.vars:
            v.optimize(lr)
        self.var_b.optimize(lr)

    def parameters(self):
        return self.vars + [self.var_b]