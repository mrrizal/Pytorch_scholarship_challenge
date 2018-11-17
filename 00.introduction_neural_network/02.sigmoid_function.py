import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def line_equation(x1, x2):
    return 4 * x1 + 5 * x2 - 9


if __name__ == '__main__':
    data = [(1, 1), (2, 4), (5, -5), (-4, 5)]
    for i in data:
    	result = line_equation(i[0], i[1])
    	result = sigmoid(result)
    	print(result)