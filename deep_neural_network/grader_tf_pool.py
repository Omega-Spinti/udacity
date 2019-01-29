import tensorflow as tf
import numpy as np
import json

result = ''


def solution(X):
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#max_pool
    return tf.nn.max_pool(X, ksize, strides, padding)


def get_result(out):
    result = {'is_correct': False, 'error': False, 'values': [],
              'output': '', 'feedback': '', 'comment': ""}

    X = tf.constant(np.random.randn(1, 4, 4, 1), dtype=tf.float32)
    ours = solution(X)
    theirs = out
    dim_names = ['Batch', 'Height', 'Width', 'Depth']

    with tf.Session() as sess:
        our_shape = ours.get_shape().as_list()
        their_shape = theirs.get_shape().as_list()

        did_pass = False

        try:
            for dn, ov, tv in zip(dim_names, our_shape, their_shape):
                if ov != tv:
                    # dimension mismatch
                    feedback = '%s dimension: mismatch we have %s, you have %s' % (dn, ov, tv)
            if np.alltrue(our_shape == their_shape):
                did_pass = True
            else:
                did_pass = False
        except:
            did_pass = False

        if did_pass:
            feedback = 'Great Job! Your output shape is: %s' % their_shape
        else:
            feedback = 'Incorrect! Correct shape is: %s Your output shape is:  %s' %(our_shape, their_shape)

    result.update(feedback=feedback)
    return result


def run_grader(out):
    try:
        # Get grade result information
        result = get_result(out)
    except Exception as err:
        # Default error result
        result = {
            'correct': False,
            'feedback': 'Something went wrong with your submission:',
            'comment': str(err)}

    feedback = result.get('feedback')
    comment = result.get('comment')

    print("%s\n%s" % (feedback,comment))


if __name__ == "__main__":
    run_grader(out)

