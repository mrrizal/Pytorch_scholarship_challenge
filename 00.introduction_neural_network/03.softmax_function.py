import numpy as np


# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    result = [np.exp(i) / sum(np.exp(L)) for i in L]
    print(result)


if __name__ == '__main__':
    mylist = [5, 6, 7]
    softmax(mylist)