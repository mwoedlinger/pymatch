from .tensor import Tensor

def mseLoss(pred: list, label: list, network=None, l2=False, l=0.0001):
    loss = Tensor(0)

    for n in range(len(pred)):
        loss = loss + (pred[n] - label[n])**2
    loss = loss*(1/len(pred))

    if l2:
        for p in network.parameters():
            loss = loss + l * p**2

    return loss

