import numpy as np


def softmax(a, tau=1):
    if tau == float('inf'):
        r = np.zeros(len(a))
        r[np.argmax(a)] = 1
        return r
    exp_a = np.exp(a * tau)
    return exp_a / exp_a.sum()
