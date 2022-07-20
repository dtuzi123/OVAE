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


class BasicBlock_Deep(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(BasicBlock_Deep, self).__init__(**kwargs)

        self.l1 = tf.keras.layers.Dense(256, activation=tf.nn.tanh)
        self.l2 = tf.keras.layers.Dense(256, activation=tf.nn.tanh)
        self.l3 = tf.keras.layers.Dense(512, activation=tf.nn.tanh)
        self.lmu = tf.keras.layers.Dense(n_latent, activation=None)
        self.lstd = tf.keras.layers.Dense(n_latent, activation=tf.exp)

    def call(self, input):
        h1 = self.l1(input)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        q_mu = self.lmu(h3)
        q_std = self.lstd(h3)

        qz_given_input = tfd.Normal(q_mu, q_std + 1e-6)

        return qz_given_input


class Encoder_Deep(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(Encoder_Deep, self).__init__(**kwargs)

        self.encode_x_to_z = BasicBlock_Deep(n_hidden, n_latent)

    def call(self, x, n_samples):
        qzx = self.encode_x_to_z(x)

        z = qzx.sample(n_samples)

        return z, qzx


class Decoder_Deep(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 **kwargs):
        super(Decoder_Deep, self).__init__(**kwargs)

        self.decode_z_to_x = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation=tf.nn.tanh),
                tf.keras.layers.Dense(256, activation=tf.nn.tanh),
                tf.keras.layers.Dense(512, activation=tf.nn.tanh),
                tf.keras.layers.Dense(784, activation=None,
                                      bias_initializer=utils.get_bias())
            ]
        )

    def call(self, z):

        logits = self.decode_z_to_x(z)

        pxz = tfd.Bernoulli(logits=logits)

        return logits, pxz


class IWAE_Deep(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(IWAE_Deep, self).__init__(**kwargs)

        self.encoder = Encoder_Deep(n_hidden, n_latent)
        self.decoder = Decoder_Deep(n_hidden)

    def call(self, x, n_samples, beta=1.0):
        # ---- encode/decode
        z, qzx  = self.encoder(x, n_samples)

        logits, pxz = self.decoder(z)

        # ---- loss
        pz = tfd.Normal(0, 1)

        lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

        lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

        lpxz = tf.reduce_sum(pxz.log_prob(x), axis=-1)

        log_w = lpxz + beta * (lpz - lqzx)

        # ---- regular VAE elbos
        kl = tf.reduce_sum(tfd.kl_divergence(qzx, pz), axis=-1)
        kl2 = -tf.reduce_mean(lpz - lqzx, axis=0)

        # mean over samples and batch
        vae_elbo = tf.reduce_mean(tf.reduce_mean(log_w, axis=0), axis=-1)
        vae_elbo_kl = tf.reduce_mean(lpxz) - beta * tf.reduce_mean(kl)

        # ---- IWAE elbos
        # eq (8): logmeanexp over samples and mean over batch
        iwae_elbo = tf.reduce_mean(utils.logmeanexp(log_w, axis=0), axis=-1)

        # eq (14):
        m = tf.reduce_max(log_w, axis=0, keepdims=True)
        log_w_minus_max = log_w - m
        w = tf.exp(log_w_minus_max)
        w_normalized = w / tf.reduce_sum(w, axis=0, keepdims=True)
        w_normalized_stopped = tf.stop_gradient(w_normalized)

        iwae_eq14 = tf.reduce_mean(tf.reduce_sum(w_normalized_stopped * log_w, axis=0))

        # ---- self-normalized importance sampling
        al = tf.nn.softmax(log_w, axis=0)

        snis_z = tf.reduce_sum(al[:, :, None] * z, axis=0)

        return {"vae_elbo": vae_elbo,
                "vae_elbo_kl": vae_elbo_kl,
                "iwae_elbo": iwae_elbo,
                "iwae_eq14": iwae_eq14,
                "z": z,
                "snis_z": snis_z,
                "al": al,
                "logits": logits,
                "lpxz": lpxz,
                "lpz": lpz,
                "lqzx": lqzx}

    @tf.function
    def train_step(self, x, n_samples, beta, optimizer, objective="vae_elbo"):
        with tf.GradientTape() as tape:
            res = self.call(x, n_samples, beta)
            loss = -res[objective]

        grads = tape.gradient(loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return res

    @tf.function
    def val_step(self, x, n_samples, beta):
        return self.call(x, n_samples, beta)

    def sample(self, z):

        logits = self.decoder.decode_z_to_x(z)

        probs = tf.nn.sigmoid(logits)

        pxz = tfd.Bernoulli(logits=logits)

        x_sample = pxz.sample()

        return x_sample, probs

    def generate_samples(self,z):
        x_samples, x_probs = self.sample(z)
        return x_samples

    def generate_and_save_images(self, z, epoch, string):

        # ---- samples from the prior
        x_samples, x_probs = self.sample(z)
        x_samples = x_samples.numpy().squeeze()
        x_probs = x_probs.numpy().squeeze()

        n = int(np.sqrt(x_samples.shape[0]))

        canvas = np.zeros((n * 28, 2 * n * 28))

        for i in range(n):
            for j in range(n):
                canvas[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = x_samples[i * n + j].reshape(28, 28)
                canvas[i * 28: (i + 1) * 28, n * 28 + j * 28: n * 28 + (j + 1) * 28] = x_probs[i * n + j].reshape(28,
                                                                                                                  28)
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.imshow(canvas, cmap='gray_r')
        plt.title("epoch {:04d}".format(epoch), fontsize=50)
        plt.axis('off')
        plt.savefig('./results/' + string + '_image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

    def generate_and_save_posteriors(self, x, y, n_samples, epoch, string):

        # ---- posterior snis means
        res = self.call(x, n_samples)

        snis_z = res["snis_z"]

        # pca
        pca = PCA(n_components=2)
        pca.fit(snis_z)
        z = pca.transform(snis_z)

        plt.clf()
        for c in np.unique(y):
            plt.scatter(z[y == c, 0], z[y == c, 1], s=10, label=str(c))
        plt.legend(loc=(1.04,0))
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.title("epoch {:04d}".format(epoch), fontsize=50)
        plt.savefig('./results/' + string + '_posterior_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

    @staticmethod
    def write_to_tensorboard(res, step):
        tf.summary.scalar('Evaluation/vae_elbo', res["vae_elbo"], step=step)
        tf.summary.scalar('Evaluation/iwae_elbo', res["iwae_elbo"], step=step)
        tf.summary.scalar('Evaluation/lpxz', res['lpxz'].numpy().mean(), step=step)
        tf.summary.scalar('Evaluation/lqzx', res['lqzx'].numpy().mean(), step=step)
        tf.summary.scalar('Evaluation/lpz', res['lpz'].numpy().mean(), step=step)




