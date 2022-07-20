import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # needed when running from commandline
import matplotlib.pyplot as plt
import cycler
import utils


# ---- plot settings
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 15.0
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['figure.autolayout'] = True
color = plt.cm.viridis(np.linspace(0, 1, 10))
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)


class TaskClassifier(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 **kwargs):
        super(TaskClassifier, self).__init__(**kwargs)

        n_hidden1 = 500
        n_hidden2 = 256
        self.classifier = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_hidden1, activation=tf.nn.tanh),
                tf.keras.layers.Dense(n_hidden2, activation=tf.nn.tanh),
                tf.keras.layers.Dense(2)  # input即为上一层的output，故定义output_dim是10个feature就可以
            ]
        )

    def call(self, x):

        logits = self.classifier(x)
        softmaxPre = tf.nn.softmax(logits)

        return logits,softmaxPre

    @tf.function
    def train_step(self, x, y, beta, optimizer, objective="vae_elbo"):
        with tf.GradientTape() as tape:
            logits,softmaxPred = self.call(x)
            loss = tf.keras.losses.binary_crossentropy(y,logits)

        grads = tape.gradient(loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss






