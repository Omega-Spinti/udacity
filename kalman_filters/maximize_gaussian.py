from math import *

# Maximize Gaussian
def f(mu, sigma2, x):
    return 1/sqrt(2.*pi*sigma2) * exp(-.5 * (x-mu)**2 / sigma2)

print(f(10., 4., 8.))

# New Mean and Variance
def update(mean1, var1, mean2, var2):
    new_mean = (var2 * mean1 + var1 * mean2) / ( var1 + var2)
    new_var = 1 / (1/var1 + 1/var2)
    return [new_mean, new_var]

print(update(10., 8., 13., 2.))

# Predict
def predict(mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]


print(predict(10., 4., 12., 4.))

# Kalman Filter Code
measurements = [5., 6., 7., 9., 10.,]
motion = [1., 1., 2., 1., 1.]
measurements_sig = 4.
motion_sig = 2.
mu = 0.
sig = 1000.


for n in range(len(measurements)):
    [mu, sig] = update(mu, sig, measurements[n], measurements_sig)
    print('update: ', [mu, sig])
    [mu, sig] = predict(mu, sig, motion[n], motion_sig)
    print('predict: ', [mu, sig])

