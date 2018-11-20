import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def error_formula(y, output):
    return -y * np.log(output) - (1 - y) * np.log(1 - output)


def error_formula_l1(y, output, weights):
    lambdanya = 1
    return (-y * np.log(output) - (1 - y) * np.log(1 - output)) + (
        lambdanya * np.sum(np.absolute(weights)))


def error_formula_l2(y, output, weights):
    lambdanya = 1
    return (-y * np.log(output) - (1 - y) * np.log(1 - output)) + (
        lambdanya * np.sum(np.power(weights, 2)))


def predict(weights, x, bias):
    return sigmoid(np.dot(weights, x) + bias)


if __name__ == '__main__':
    x = np.array([[1, 1], [-1, -1]]).T
    bias = 0
    weights = np.array([10, 10])
    output = predict(weights, x, bias)
    print(error_formula(np.array([1, 0]), output))
    print(error_formula_l1(np.array([1, 0]), output, weights))
    print(error_formula_l2(np.array([1, 0]), output, weights))

# dikenal juga sebagai laso regression
# l1 regularization = better for feature selection
# l1 bagi digunakan ketika most of variable  not usefull

# dikenal juga sebagai ridge regression
# l2 regularization = normally better for training models
# l2 baik digunakan ketika most of variable/feature is usefull

# lambda merupakan angka 0 sampai inifity positif, disini kita asumsikan nilai lambda adalah 1
# untuk tunning parameter bisa gunakan grid search
# https://scikit-learn.org/stable/modules/grid_search.html

# videos:
# https://www.youtube.com/watch?v=Q81RR3yKn30 (ridge regression/l2)
# https://www.youtube.com/watch?v=NGf0voTMlcs (laso regression/l1)
# https://www.youtube.com/watch?v=1dKRdX9bfIo (elastic net regression -> kombinasi ridge dan laso regression)
