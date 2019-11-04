import numpy as np


# sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1

#make a loop over 10 000 iteration to update the weights for a ONE LAYER neural network with 1 ouput
#in the loop you should start by evaluating the output with the current weights
#then you compute the error made
#to update your weights, you can use the 'gradient method'
#finally you need to update your weights

