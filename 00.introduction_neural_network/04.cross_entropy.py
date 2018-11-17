import numpy as np


def cross_entropy(Y, P):
    result = 0
    for i in range(len(Y)):
        result += Y[i] * -np.log(P[i]) + (1 - Y[i]) * -np.log(1 - P[i])

    return result


if __name__ == '__main__':
    Y = [1, 0, 1, 1]
    P = [0.4, 0.6, 0.1, 0.5]
    result = cross_entropy(Y, P)
    print(result)