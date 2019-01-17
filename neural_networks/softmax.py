import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    expL = np.exp(L)
    sum_expL = sum(expL)

    result = []

    for i in expL:
        result.append(i * 1.0/sum_expL)

    print(result)
    return result


L = [-1,0,1]
softmax(L)

# Note: The function np.divide can also be used here, as follows:
# def softmax(L):
#     expL = np.exp(L)
#     return np.divide (expL, expL.sum())