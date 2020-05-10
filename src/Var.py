class Var:
    def __init__(self, val, name=''):
        self.val = val
        self.cMult = []
        self.cAdd = []
        self.cPow = {}
        self.zero_grad()
        self.no_grad = False
        self.name = name

    def __mul__(self, x):
        if not isinstance(x, Var):
            x = Var(x, self.name+'_mulInt')

        out = Var(self.val * x.val, self.name+'_mult')
        out.cMult = [self, x]
        return out

    def __add__(self, x):
        if not isinstance(x, Var):
            x = Var(x, self.name+'_addIn')
 
        out = Var(self.val + x.val, self.name+'_add')
        out.cAdd = [self, x]

        return out

    def __pow__(self, exponent):
        out = Var(self.val**exponent, self.name+'_pow')
        out.cPow = {self: exponent}

        return out

    def __radd__(self, x):
        return Var(x, self.name+'_raddIn') + self

    def __neg__(self):
        return self*(-1)

    def __sub__(self, x):
        return self + (-x)

    def __rsub__(self, x):
        return Var(x, self.name+'_rsubIn') - self

    def __repr__(self):
        return str(self.val)

    def __rmul__(self, x):
        return Var(x, self.name+'_rmulIn') * self

    def zero_grad(self):
        self.grad = 0

    def backward(self):
        if self.no_grad:
            return

        for c in self.cAdd:
            gradBuff = c.grad
            c.grad = self.grad
            c.backward()
            c.grad += gradBuff

        if self.cMult:
            gradBuff = self.cMult[0].grad
            self.cMult[0].grad = self.cMult[1].val * self.grad
            self.cMult[0].backward()
            self.cMult[0].grad += gradBuff

            gradBuff = self.cMult[1].grad
            self.cMult[1].grad = self.cMult[0].val * self.grad
            self.cMult[1].backward()
            self.cMult[1].grad += gradBuff
        
        if self.cPow:
            for x in self.cPow.keys():
                gradBuff = x.grad
                x.grad = self.cPow[x]*x.val 
                x.backward()
                x.grad += gradBuff

    def optimize(self, lr):
        if not self.no_grad:
            self.val -= lr * self.grad