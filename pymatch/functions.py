import numpy as np
from .variable import Variable


# Functions on Variables:
def relu(x: Variable):
    if not isinstance(x, Variable):
        x = Variable(x, name=str(x))

    out = Variable(max(x.val, 0), 
                 children = [x],
                 name='relu({})'.format(x.name))
    out.children = [x]

    def _computeGradient():
        if x.val > 0:
            x.grad += out.grad
        else:
            x.grad = 0
    out._computeGradient = _computeGradient

    return out


def log(x: Variable):
    if not isinstance(x, Variable):
        x = Variable(x, name=str(x))

    out = Variable(np.log(x.val), 
                 children = [x],
                 name='log({})'.format(x.name))
    out.children = [x]

    def _computeGradient():
        x.grad += out.grad / x.val
    out._computeGradient = _computeGradient

    return out

def abs(x: Variable):
    if not isinstance(x, Variable):
        x = Variable(x, name=str(x))

    out = Variable(np.abs(x.val), 
                 children = [x],
                 name='abs({})'.format(x.name))
    out.children = [x]

    def _computeGradient():
        if x.val > 0:
            x.grad += out.grad
        else:
            x.grad = -out.grad
    out._computeGradient = _computeGradient

    return out

def exp(x: Variable):
    if not isinstance(x, Variable):
        x = Variable(x, name=str(x))

    out = Variable(np.abs(x.val), 
                 children = [x],
                 name='exp({})'.format(x.name))
    out.children = [x]

    def _computeGradient():
        x.grad += out.grad * out.val
    out._computeGradient = _computeGradient

    return out

def softmax(x: list):
    out = [exp(z) * (sum([exp(y) for y in x]))**(-1) for z in x]

    return out

def sigmoid(x: Variable):
    out = exp(x) * (exp(x) + 1)**(-1)

    return out

# Loss functions:
def mseLoss(pred: list, label: list, network, l1=False, l2=False, alpha=0.0001):
    loss = sum([(pred[n] - label[n])**2 
                for n in range(len(pred))]) * (1/len(pred))
    if l1:
        loss = loss + alpha * sum([abs(p) for p in network.parameters()])
    if l2:
        loss = loss + alpha * sum([p**2 for p in network.parameters()])

    return loss

def crossentropyLoss(pred: list, label: list, network, l1=False, l2=False, alpha=0.0001):
    loss = -sum([label[n]*log(pred[n]) for n in range(len(pred))])

    if l1:
        loss = loss + alpha * sum([abs(p) for p in network.parameters()])
    if l2:
        loss = loss + alpha * sum([p**2 for p in network.parameters()])

    return loss

def maxMarginLoss(pred, label, network, l1=False, l2=False, alpha=0.0001):
    loss = relu(1 - label[0] * pred[0])

    if l1:
        loss = loss + alpha * sum([abs(p) for p in network.parameters()])
    if l2:
        loss = loss + alpha * sum([p**2 for p in network.parameters()])

    return loss


    