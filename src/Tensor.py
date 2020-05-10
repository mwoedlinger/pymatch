class Tensor:
    def __init__(self, val, name='', op=''):
        self.name = name
        self.val = val
        self.op = op

        self.children = []
        
        self.no_grad = False
        self.grad = 0
        self.depth = 0

    def __mul__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, name=str(x))

        out = Tensor(self.val * x.val,
                     name='prod({}, {})'.format(self.name, x.name), 
                     op='*')
        out.children = [self, x]

        return out

    def __add__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, name=str(x))

        out = Tensor(self.val + x.val,
                     name='sum({}, {})'.format(self.name, x.name), 
                     op='+')
        out.children = [self, x]
        
        return out

    def __pow__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, name=str(x))

        out = Tensor(self.val**x.val,
                     name='pow({}, {})'.format(self.name, x.name), 
                     op='^')
        out.children = [self, x]
        
        return out

    def relu(self):
        out = Tensor(max(self.val, 0), 
                     'relu({})'.format(self.name),
                     op='relu')
        out.children = [self]

        return out

    def __neg__(self):
        return self*(-1)

    def __radd__(self, x):
        return Tensor(x) + self

    def __sub__(self, x):
        return self + (-x)

    def __rsub__(self, x):
        return Tensor(x) - self

    def __rmul__(self, x):
        return Tensor(x) * self
        
    def __repr__(self):
        return str(self.val)

    def _depthCount(self, count):
        self.depth = max(self.depth, count+1)

        maxDepth = self.depth

        for c in self.children:
            maxDepth = max(maxDepth, c._depthCount(self.depth))

        return maxDepth 
    
    def _updateDepthDict(self, dDict):
        dDict[self.depth].add(self)

        for c in self.children:
            c._updateDepthDict(dDict)


    def depthDict(self):
        self.depth = 0
        dDict = {n: set() for n in range(self._depthCount(0) + 1)}
        self._updateDepthDict(dDict)

        return dDict

    def _computeGradient(self):
        if self.no_grad:
            self.grad = 0
            return

        if self.op == '+':
            for c in self.children:
                c.grad += self.grad

        elif self.op == '*':
            self.children[0].grad += self.grad * self.children[1].val
            self.children[1].grad += self.grad * self.children[0].val

        elif self.op == '^':
            base = self.children[0]
            exponent = self.children[1]

            base.grad += exponent.val * base.val**(exponent.val - 1)

        elif self.op == 'relu':
            if self.children[0].val > 0:
                self.children[0].grad += self.grad
            else:
                self.children[0].grad = 0

    def zero_grad(self):
        self.grad = 0
        
        for c in self.children:
            c.zero_grad()

    def backward(self, dDict):
        self.grad = 1
        maxDepth = max(dDict.keys())

        for d in range(0, maxDepth+1):
            for t in dDict[d]:
                t._computeGradient()

    def _optimize(self, lr):
        if not self.no_grad:
            self.val -= lr * self.grad

    def optimize(self, lr, dDict):
        # TODO: move outside of class as function
        maxDepth = max(dDict.keys())

        for d in range(0, maxDepth+1):
            for t in dDict[d]:
                t._optimize(lr)
