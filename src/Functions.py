from .tensor import Tensor

def mseLoss(pred: list, label: list):
    loss = Tensor(0)

    for n in range(len(pred)):
        loss = loss + (pred[n] - label[n])**2
    loss = loss*(1/len(pred))

    return loss