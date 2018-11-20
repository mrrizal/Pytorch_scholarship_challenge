# quiz, which one gives a smaller error
# weights = [1,1] atau weights = [10,10]
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def error_formula(y, output):
    return -y * np.log(output) - (1 - y) * np.log(1 - output)


def predict(weights, x, bias):
    return sigmoid(np.dot(weights, x) + bias)


if __name__ == '__main__':
    x = np.array([[1, 1], [-1, -1]]).T
    bias = 0
    weights = np.array([1, 1])
    output = predict(weights, x, bias)
    print(output)
    print(error_formula(np.array([1, 0]), output))

    weights = np.array([10, 10])
    output = predict(weights, x, bias)
    print(output)
    print(error_formula(np.array([1, 0]), output))
