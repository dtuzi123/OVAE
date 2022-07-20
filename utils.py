import tensorflow as tf
import numpy as np
from tensorflow import keras


def logmeanexp(log_w, axis):
    max = tf.reduce_max(log_w, axis=axis)
    return tf.math.log(tf.reduce_mean(tf.exp(log_w - max), axis=axis)) + max


def get_bias():
    # ---- For initializing the bias in the final Bernoulli layer for p(x|z)
    (Xtrain, ytrain), (_, _) = keras.datasets.mnist.load_data()
    Ntrain = Xtrain.shape[0]

    # ---- reshape to vectors
    Xtrain = Xtrain.reshape(Ntrain, -1) / 255

    train_mean = np.mean(Xtrain, axis=0)

    bias = -np.log(1. / np.clip(train_mean, 0.001, 0.999) - 1.)

    return tf.constant_initializer(bias)


def bernoullisample(x):
    return np.random.binomial(1, x, size=x.shape).astype('float32')


class MyMetric():
    def __init__(self):
        self.VALUES = []
        self.N = []

    def update_state(self, losses):
        self.VALUES.append(losses)
        self.N.append(losses.shape[0])

    def result(self):
        VALUES = tf.concat(self.VALUES, axis=0)
        return tf.reduce_sum(VALUES) / tf.cast(tf.reduce_sum(self.N), tf.float32)

    def reset_states(self):
        self.VALUES = []
        self.N = []
