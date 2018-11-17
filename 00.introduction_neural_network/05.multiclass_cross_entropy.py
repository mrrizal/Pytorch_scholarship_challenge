import numpy as np


def multiclass_cross_entropy(Y, P):
    result = 0
    for i in range(len(Y)):
        temp = 0
        for j in range(len(Y[i])):
            # print(Y[i][j], P[i][j])
            temp += Y[i][j] * -np.log(P[i][j])

        result += temp

    return result


if __name__ == '__main__':
    Y = [[1, 0, 0], [0, 0, 0], [0, 1, 1]]
    P = [[0.7, 0.3, 0.1], [0.2, 0.4, 0.5], [0.1, 0.3, 0.4]]
    result = multiclass_cross_entropy(Y, P)
    print(result)