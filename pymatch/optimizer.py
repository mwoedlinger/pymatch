
def sgd(lr, loss):
    maxDepth = max(loss.dDict.keys())

    for d in range(0, maxDepth+1):
        for t in loss.dDict[d]:
            if not t.no_grad:
                t.val -= lr * t.grad