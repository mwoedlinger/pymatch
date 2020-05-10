import math
from .Var import Var

# Activation functions:
class ReLU:
    def __call__(self, x):
        if x.val > 0:
            return x
        else:
            return Var(0, 'ReLU')

    def backward_call(self, x):
        if x > 0:
            return 1
        else:
            return 0

class Identity:
    def __call__(self, x):
        return x

    def backward_call(self, x):
        return 1

# Loss functions:
class MSELoss:
    def __init__(self, data, network, label, l2=False, l=0.001):
        network.zero_grad()
        pred = network(data)
        
        loss = 0

        for n in range(len(pred)):
            loss = loss + (pred[n] -label[n])**2
        # TODO: divide by len(pred)

        if l2:
            for p in network.parameters():
                loss += l*p**2

        loss.grad = 1

        self.pred = pred
        self.loss = loss
        self.network = network

    def backward(self):
        self.loss.backward()


# class CrossEntropyLoss:
#     def __init__(self, data, network, label, l2=True, l=0.001):
#         network.zero_grad()
#         pred = network(data)
        
#         loss = 0

#         for yhat in pred:
#             loss = loss + ()





#         loss = 0

#         for n in range(len(pred)):
#             loss = loss + (pred[n] -label[n])**2

#         if l2:
#             for p in network.parameters():
#                 loss += l*p**2

#         loss.grad = 1

#         self.pred = pred
#         self.loss = loss
#         self.network = network

#     def backward(self):
#         self.loss.backward()
