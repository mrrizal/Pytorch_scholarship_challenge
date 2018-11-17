import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def equation(w1, w2, bias):
    features = np.array([0.4, 0.6])
    weights = np.array([w1, w2])
    return sigmoid(np.dot(weights, features) + bias)


if __name__ == '__main__':
    parameters = [{
        "w1": 2,
        "w2": 6,
        "b": -2
    }, {
        "w1": 3,
        "w2": 5,
        "b": -2.2
    }, {
        "w1": 5,
        "w2": 4,
        "b": -3
    }]

    for parameter in parameters:
        result = equation(parameter["w1"], parameter["w2"], parameter["b"])
        print(result)
