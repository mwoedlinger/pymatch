import random

class DataLoader:
    def __init__(self, X, Y, batchSize=1, shuffle=False):
        self.data = [(X[n], Y[n]) for n in range(len(X))]
        self.batchSize = batchSize
        self.shuffleData = shuffle
        self.idx = 0

    def __len__(self):
        return round(len(self.data))

    def __iter__(self):
        return self

    def __next__(self):
        if (self.idx+1) * self.batchSize >= len(self.data):
            self.idx = 0
            if self.shuffle:
                self.shuffle()
            raise StopIteration
        else:
            self.idx += 1
            return self.data[self.idx*self.batchSize:(self.idx+1)*self.batchSize]

    def shuffle(self):
        random.shuffle(self.data)