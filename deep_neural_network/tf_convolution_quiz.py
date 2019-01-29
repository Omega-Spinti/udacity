import tensorflow as tf
import numpy as np

"""
Setup the strides, padding and filter weight/bias such that
the output shape is (1, 2, 2, 3).
"""
# `tf.nn.conv2d` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)

def conv2d(input):
    # Filter (weights and bias)
    F_W = tf.Variable(tf.truncated_normal((2, 2, 1, 3)))
    F_b = tf.Variable(tf.zeros(3))
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    return tf.nn.conv2d(input, F_W, strides, padding) + F_b

output = conv2d(X)
print(output)

##### Do Not Modify ######

import grader_tf_cnn as grader

test_X = tf.constant(np.random.randn(1, 4, 4, 1), dtype=tf.float32)

try:
    response = grader.run_grader(test_X, conv2d)
    print(response)


except Exception as err:
    print(str(err))
