class Tensor:
    def __init__(self, val=0, children=[], name=''):
        self.name = name
        self.val = val

        self.children = children
        self._computeGradient = lambda: None
        
        self.no_grad = False
        self.grad = 0

        if children:
            self.depth = max([c.depth for c in children]) + 1
        else:
            self.depth = 0

        self.dDict = {n : set() for n in range(0, self.depth)}
        self.dDict.update({self.depth : set([self])})
        for c in children:
            for k in c.dDict.keys():
                self.dDict[k].update(c.dDict[k])

    def __mul__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, name=str(x))

        out = Tensor(self.val * x.val,
                     children = [self, x],
                     name='prod({}, {})'.format(self.name, x.name))

        def _computeGradient():
            self.grad += out.grad * x.val
            x.grad += out.grad * self.val
        out._computeGradient = _computeGradient

        return out

    def __add__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, name=str(x))

        out = Tensor(self.val + x.val,
                     children = [self, x],
                     name='sum({}, {})'.format(self.name, x.name))
        
        def _computeGradient():
            self.grad += out.grad
            x.grad += out.grad
        out._computeGradient = _computeGradient
        
        return out

    def __pow__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, name=str(x))

        out = Tensor(self.val**x.val,
                     children = [self, x],
                     name='pow({}, {})'.format(self.name, x.name))

        def _computeGradient():
            self.grad += out.grad * x.val * self.val**(x.val - 1)
        out._computeGradient = _computeGradient
        
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
        return str(self.name)

    def zero_grad(self):
        self.grad = 0
        
        for c in self.children:
            c.zero_grad()

    def backward(self, grad=1):
        self.grad = grad

        maxDepth = max(self.dDict.keys())
        minDepth = min(self.dDict.keys())

        for d in range(maxDepth, minDepth-1, -1):
            for t in self.dDict[d]:
                t._computeGradient()





# class Tensor:
#     def __init__(self, val=0, children=[], name=''):
#         self.name = name
#         self.val = val

#         self.children = children
#         self._computeGradient = lambda: None
        
#         self.no_grad = False
#         self.grad = 0

#         self.dDict = None
#         self.depth = 0    

#     def __mul__(self, x):
#         if not isinstance(x, Tensor):
#             x = Tensor(x, name=str(x))

#         out = Tensor(self.val * x.val,
#                      children = [self, x],
#                      name='prod({}, {})'.format(self.name, x.name))

#         def _computeGradient():
#             self.grad += out.grad * x.val
#             x.grad += out.grad * self.val
#         out._computeGradient = _computeGradient

#         return out

#     def __add__(self, x):
#         if not isinstance(x, Tensor):
#             x = Tensor(x, name=str(x))

#         out = Tensor(self.val + x.val,
#                      children = [self, x],
#                      name='sum({}, {})'.format(self.name, x.name))
        
#         def _computeGradient():
#             self.grad += out.grad
#             x.grad += out.grad
#         out._computeGradient = _computeGradient
        
#         return out

#     def __pow__(self, x):
#         if not isinstance(x, Tensor):
#             x = Tensor(x, name=str(x))

#         out = Tensor(self.val**x.val,
#                      children = [self, x],
#                      name='pow({}, {})'.format(self.name, x.name))

#         def _computeGradient():
#             self.grad += out.grad * x.val * self.val**(x.val - 1)
#         out._computeGradient = _computeGradient
        
#         return out

#     def __neg__(self):
#         return self*(-1)

#     def __radd__(self, x):
#         return Tensor(x) + self

#     def __sub__(self, x):
#         return self + (-x)

#     def __rsub__(self, x):
#         return Tensor(x) - self

#     def __rmul__(self, x):
#         return Tensor(x) * self
        
#     def __repr__(self):
#         return str(self.val)

#     def _depthCount(self, count):
#         self.depth = max(self.depth, count+1)

#         maxDepth = self.depth

#         for c in self.children:
#             maxDepth = max(maxDepth, c._depthCount(self.depth))

#         return maxDepth 
    
#     def _updateDepthDict(self, dDict):
#         dDict[self.depth].add(self)

#         for c in self.children:
#             c._updateDepthDict(dDict)


#     def _depthDict(self):
#         self.depth = -1
#         dDict = {n: set() for n in range(self._depthCount(self.depth) + 1)}
#         self._updateDepthDict(dDict)

#         return dDict

#     def resetDepthDict(self):
#         self.dDict = None

#     def zero_grad(self):
#         self.grad = 0
        
#         for c in self.children:
#             c.zero_grad()

#     def backward(self, grad=1):
#         self.grad = grad

#         if self.dDict == None:
#             self.dDict = self._depthDict()

#         maxDepth = max(self.dDict.keys())

#         for d in range(0, maxDepth+1):
#             for t in self.dDict[d]:
#                 t._computeGradient()
