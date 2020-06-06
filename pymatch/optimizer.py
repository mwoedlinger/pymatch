from math import sqrt


def sgd(loss, lr, momentum=0.9):
    """
    Performs gradient decent.
    """
    max_depth = max(loss.d_dict.keys())

    for d in range(0, max_depth+1):
        for t in loss.d_dict[d]:
            if not t.no_grad:                
                if hasattr(t, 'v'):
                    t.v = momentum * t.v - lr * t.grad
                else:
                    t.v = -lr * t.grad

                t.val += t.v

def rms_prob(loss, lr, decay=0.9, eps=0.00000001):
    """
    Performs gradient decent.
    """
    max_depth = max(loss.d_dict.keys())

    for d in range(0, max_depth+1):
        for t in loss.d_dict[d]:
            if not t.no_grad:                
                if hasattr(t, 'cache'):
                    t.cache = decay * t.cache + (1 - decay) * t.grad**2
                else:
                    t.cache = (1 - decay) * t.grad**2

                t.val += -lr * t.grad / (sqrt(t.cache) + eps)