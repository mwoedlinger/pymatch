import numpy as np
import matplotlib.pyplot as plt

from src.Functions import ReLU, Identity, MSELoss
from src.Layer import Layer
from src.Network import Network
from src.Neuron import Neuron
from src.Var import Var

if __name__ == '__main__':
    x = [2, 3]
    y = [5, 7]

    l0 = Layer(2, 5, ReLU(), 'l0')
    l1 = Layer(5, 5, ReLU(), 'l1')
    l2 = Layer(5, 2, Identity(), 'l2')

    n = Network([l0, l1, l2])

    epochs = 50
    lr = 0.1

    for epoch in range(epochs):
        mse = MSELoss(x, n, y)
        print('Epoch: {}, Loss = {}, n(x) = {}'.format(epoch, mse.loss, mse.pred))

        n.zero_grad()
        mse.backward()
        n.optimize(lr)

    print('n(x) = {}'.format(n(x)))