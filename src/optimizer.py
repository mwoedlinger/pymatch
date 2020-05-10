
def sgd(lr, dDict):
    maxDepth = max(dDict.keys())

    for d in range(0, maxDepth+1):
        for t in dDict[d]:
            if not t.no_grad:
                t.val -= lr * t.grad