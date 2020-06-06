class Variable:
    """
    Variable objects keep track of the the computation graph in self.d_dict and their gradient in self.grad.
    """
    def __init__(self, val=0, children=[], name=''):
        self.name = name
        self.val = val

        self.children = children
        self._computeGradient = lambda: None
        
        self.no_grad = False
        self.grad = 0

        # self.depth tells how far the current variable is aways from the input layer.
        if children:
            self.depth = max([c.depth for c in children]) + 1
        else:
            self.depth = 0

        # self.d_dict is a dictionary that has depth values as keys and gives
        # for every depth every variable that self is dependent on
        self.d_dict = {n : set() for n in range(0, self.depth)}
        self.d_dict.update({self.depth : set([self])})
        for c in children:
            for k in c.d_dict.keys():
                self.d_dict[k].update(c.d_dict[k])

    def __mul__(self, x):
        if not isinstance(x, Variable):
            x = Variable(x, name=str(x))

        out = Variable(self.val * x.val,
                     children = [self, x],
                     name='prod({}, {})'.format(self.name, x.name))

        def _computeGradient():
            self.grad += out.grad * x.val
            x.grad += out.grad * self.val
        out._computeGradient = _computeGradient

        return out

    def __add__(self, x):
        if not isinstance(x, Variable):
            x = Variable(x, name=str(x))

        out = Variable(self.val + x.val,
                     children = [self, x],
                     name='sum({}, {})'.format(self.name, x.name))
        
        def _computeGradient():
            self.grad += out.grad
            x.grad += out.grad
        out._computeGradient = _computeGradient
        
        return out

    def __pow__(self, x):
        if not isinstance(x, Variable):
            x = Variable(x, name=str(x))

        out = Variable(self.val**x.val,
                     children = [self, x],
                     name='pow({}, {})'.format(self.name, x.name))

        def _computeGradient():
            self.grad += out.grad * x.val * self.val**(x.val - 1)
        out._computeGradient = _computeGradient
        
        return out

    def __neg__(self):
        out = self*(-1)
        out.name = '-{}'.format(out.name)

        return out

    def __radd__(self, x):
        return Variable(x) + self

    def __sub__(self, x):
        return self + (-x)

    def __rsub__(self, x):
        return Variable(x) - self

    def __rmul__(self, x):
        return Variable(x) * self
        
    def __repr__(self):
        return '{:0.4f}'.format(self.val)

    def __lt__(self, x):
        if not isinstance(x, Variable):
            x = Variable(x, name=str(x))

        return self.val < x.val

    def __le__(self, x):
        if not isinstance(x, Variable):
            x = Variable(x, name=str(x))

        return self.val <= x.val

    def __gt__(self, x):
        return -self < -x

    def __ge__(self, x):
        return -self <= -x

    def zero_grad(self):
        """
        Sets the gradient of self and all variables it depends on to zero.
        """
        self.grad = 0
        
        for c in self.children:
            c.zero_grad()

    def backward(self, grad=1):
        """
        Computes the gradients for all variables that self depends on.
        """
        self.grad = grad

        maxDepth = max(self.d_dict.keys())
        minDepth = min(self.d_dict.keys())

        for d in range(maxDepth, minDepth-1, -1):
            for t in self.d_dict[d]:
                t._computeGradient()

