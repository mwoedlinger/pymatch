from random import uniform
from .variable import Variable

class Tensor:
    """
    A Tensor is a collection of Variables. Tensors are defined recursively, i.e. The self.data attribute of a Tensor with
    shape (3,4,5) is a list of three (4,5) Tensors, each of which contains again a list of four (5, ) Tensors which contains 
    a list of five Variables.
    """
    def __init__(self, data: list, name: str=''):
        self.data = data
        self.name = name
        self._shape = self.shape()

    def shape(self):
        if hasattr(self, '_shape'):
            return self._shape
        else:
            if isinstance(self.data[0], Tensor):
                return (len(self.data), ) + self.data[0].shape()
            else:
                return (len(self.data), )

    @staticmethod
    def rand(shape: tuple, name: str='', a: int=0, b: int=1):
        """
        Returns a Tensor with random values in [a,b], sampled from a uniform distribution.
        """
        if len(shape) > 1:
            data = [Tensor.rand(shape=shape[1:], name=name+str(n), a=a, b=b) for n in range(shape[0])]
        else:
            data = [Variable(val=uniform(a, b), name=name+str(n)) for n in range(shape[0])]
        
        return Tensor(data=data, name=name)

    @staticmethod
    def zeros(shape: tuple, name: str=''):
        if len(shape) > 1:
            data = [Tensor.zeros(shape=shape[1:], name=name+str(n)) for n in range(shape[0])]
        else:
            data = [Variable(val=0, name=name+str(n)) for n in range(shape[0])]
        
        return Tensor(data=data, name=name)

    @staticmethod
    def ones(shape: tuple, name: str=''):
        if len(shape) > 1:
            data = [Tensor.ones(shape=shape[1:], name=name+str(n)) for n in range(shape[0])]
        else:
            data = [Variable(val=1, name=name+str(n)) for n in range(shape[0])]
        
        return Tensor(data=data, name=name)

    def contract(self, other):
        """
        Contracts the 1-dim Tensor 'other' in the last dimension.
        """
        assert len(other.shape()) == 1, 'Only matrix vector multiplication is allowed'
        assert other.shape() == self.shape()[-1:], 'Only matrix vector multiplication is allowed'

        if len(self._shape) > 1:
            data_out = [self.data[n].contract(other) for n in range(self._shape[0])]
            return Tensor(data=data_out, name='contract({}, {})'.format(self.name, other.name))
            #for d in data_out:
            #    d.shape = d.shape[:-1]
        else:
            return sum(self*other)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __add__(self, other):
        """
        Element wise addition.
        """
        assert other.shape() == self.shape(), 'Cannot add tensors with different shapes'

        data_out = [self.data[n] + other.data[n] for n in range(self._shape[0])]

        return Tensor(data=data_out, name='sum({}, {})'.format(self.name, other.name))
        
    def __mul__(self, other):
        """
        Element wise product
        """
        assert self.shape() == other.shape(), 'Tensors must have identical shape'

        data_out = [self.data[n] * other.data[n] for n in range(self._shape[0])]

        return Tensor(data=data_out, name='mul({}, {})'.format(self.name, other.name))

    def __repr__(self):
        if len(self._shape) > 1:
            s = '[ ' + ', '.join([str(Tensor(self.data[n].data)) for n in range(self._shape[0])]) + ' ]'
        else:
            s = '\n[ ' + ', '.join([str(self.data[n]) for n in range(self._shape[0])]) + ' ]'

        return s