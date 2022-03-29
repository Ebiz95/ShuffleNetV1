import tensorflow as tf
from tensorflow import keras

def combine_func(x, y, combine):
    if combine == 'add':
        return tf.add(x, y)
    else: # concat
        return tf.concat([x, y], axis=-1)

def channel_shuffle(inputs, groups):
    n, h, w, c = inputs.shape.as_list()
    x_reshaped = tf.reshape(inputs, [-1, h, w, groups, c // groups])
    x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
    output = tf.reshape(x_transposed, [-1, h, w, c])
    return output

def make_model():
    pass