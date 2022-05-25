from __future__ import print_function

import numpy as np
import math

def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])


def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1  # plus the 0th word


def fit_in_vocabulary(X, voc_size):
    return [[w for w in x if w < voc_size] for x in X]


def batch_generator(X, y, batch_size):
    """Primitive batch generator 
    """
    size = X.shape[0]
    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue

def batch_generator2(X1, X2, y, batch_size):
    """Primitive batch generator 
    """
    size = X1.shape[0]
    indices = np.arange(size)
    np.random.shuffle(indices)
    X1_copy = X1[indices]
    X2_copy = X2[indices]
    y_copy = y[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X1_copy[i:i + batch_size], X2_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X1_copy = X1[indices]
            X2_copy = X2[indices]
            y_copy = y[indices]
            continue

def batch_generator_multi(x1,x2, y, batch_size):
    """Primitive batch generator 
    """
    size = x1.shape[0]
    indices = np.arange(size)
    i = 0
    while i + batch_size <= size:
        yield x1[i:i + batch_size], x2[i:i + batch_size], y[i:i + batch_size]
        i += batch_size
    if size-i > 0:
        yield x1[i:size], x2[i:size], y[i:size]

def batch_generator_single(x1, y, batch_size):
    """Primitive batch generator 
    """
    size = x1.shape[0]
    indices = np.arange(size)
    i = 0
    while True:
        if i + batch_size <= size:
            yield x1[i:i + batch_size], y[i:i + batch_size]
            i += batch_size
        else:
            yield x1[i:size], y[i:size]
def batch_generator_single2(x1, y, batch_size):
    """Primitive batch generator 
    """
    size = x1.shape[0]
    indices = np.arange(size)
    i = 0
    while i + batch_size <= size:
        yield x1[i:i + batch_size], y[i:i + batch_size]
        i += batch_size
    if size-i > 0:
        yield x1[i:size], y[i:size]
if __name__ == "__main__":
    # Test batch generator
    m = np.array(['a', 'b', 'c', 'd','e'])
    n = np.array([1, 2, 3, 4, 5])
    gen = batch_generator_single2(m, n, 5)
    for i,j in gen:
        print(i,j)
    #for _ in range(math.ceil(m.shape[0]/2)):
    #    xx, yy = next(gen)
    #    print(xx, yy)
