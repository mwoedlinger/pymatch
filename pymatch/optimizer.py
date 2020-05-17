
def sgd(lr, loss):
    """
    Performs gradient decent.
    """
    max_depth = max(loss.d_dict.keys())

    for d in range(0, max_depth+1):
        for t in loss.d_dict[d]:
            if not t.no_grad:
                t.val -= lr * t.grad