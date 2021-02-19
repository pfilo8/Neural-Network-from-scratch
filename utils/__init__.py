import numpy as np


def accuracy(y, y_hat):
    return np.mean(y == y_hat)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
