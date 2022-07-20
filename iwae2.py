import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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


class BasicBlock(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(BasicBlock, self).__init__(**kwargs)

        self.l1 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.l2 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.lmu = tf.keras.layers.Dense(n_latent, activation=None)
        self.lstd = tf.keras.layers.Dense(n_latent, activation=tf.exp)

    def call(self, input):
        h1 = self.l1(input)
        h2 = self.l2(h1)
        q_mu = self.lmu(h2)
        q_std = self.lstd(h2)

        qz_given_input = tfd.Normal(q_mu, q_std + 1e-6)

        return qz_given_input


class Encoder(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.encode_x_to_z1 = BasicBlock(n_hidden[0], n_latent[0])
        self.encode_z1_to_z2 = BasicBlock(n_hidden[1], n_latent[1])

    def call(self, x, n_samples):
        qz1x = self.encode_x_to_z1(x)

        z1 = qz1x.sample(n_samples)

        qz2z1 = self.encode_z1_to_z2(z1)

        z2 = qz2z1.sample()

        return z1, qz1x, z2, qz2z1


class Decoder(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.decode_z2_to_z1 = BasicBlock(n_hidden[1], n_latent)

        # decode z1 to x
        self.decode_z1_to_x = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_hidden[0], activation=tf.nn.tanh),
                tf.keras.layers.Dense(n_hidden[0], activation=tf.nn.tanh),
                tf.keras.layers.Dense(784, activation=None,
                                      bias_initializer=utils.get_bias())
            ]
        )

    def call(self, z1, z2):
        pz1z2 = self.decode_z2_to_z1(z2)

        logits = self.decode_z1_to_x(z1)

        pxz1 = tfd.Bernoulli(logits=logits)

        return logits, pxz1, pz1z2


class IWAE(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(IWAE, self).__init__(**kwargs)

        self.encoder = Encoder(n_hidden, n_latent)
        self.decoder = Decoder(n_hidden, n_latent[0])

    def GiveReconstruction(self,x,n_samples):
        z1, qz1x, z2, qz2z1 = self.encoder(x, n_samples)
        logits, pxz1, pz1z2 = self.decoder(z1, z2)

        reco = pxz1.sample(1)
        return reco

    def Give_Inference(self,x,n_samples):
        z1, qz1x, z2, qz2z1 = self.encoder(x, n_samples)
        return qz2z1

    def call(self, x, n_samples, beta=1.0):
        # ---- encode/decode
        z1, qz1x, z2, qz2z1 = self.encoder(x, n_samples)

        logits, pxz1, pz1z2 = self.decoder(z1, z2)

        # ---- loss
        pz2 = tfd.Normal(0, 1)

        lpz2 = tf.reduce_sum(pz2.log_prob(z2), axis=-1)

        lqz2z1 = tf.reduce_sum(qz2z1.log_prob(z2), axis=-1)

        lpz1z2 = tf.reduce_sum(pz1z2.log_prob(z1), axis=-1)

        lqz1x = tf.reduce_sum(qz1x.log_prob(z1), axis=-1)

        lpxz1 = tf.reduce_sum(pxz1.log_prob(x), axis=-1)

        log_w = lpxz1 + lpz1z2 + lpz2 - lqz1x - lqz2z1

        # ---- regular VAE elbo
        # mean over samples and batch
        vae_elbo = tf.reduce_mean(tf.reduce_mean(log_w, axis=0), axis=-1)

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

        snis_z1 = tf.reduce_sum(al[:, :, None] * z1, axis=0)

        snis_z2 = tf.reduce_sum(al[:, :, None] * z2, axis=0)

        return {"vae_elbo": vae_elbo,
                "iwae_elbo": iwae_elbo,
                "iwae_eq14": iwae_eq14,
                "z1": z1,
                "z2": z2,
                "snis_z1": snis_z1,
                "snis_z2": snis_z2,
                "al": al,
                "logits": logits,
                "lpxz1": lpxz1,
                "lpz1z2": lpz1z2,
                "lpz2": lpz2,
                "lqz1x": lqz1x,
                "lqz2z1": lqz2z1}

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

    def sample(self, z2):
        pz1z2 = self.decoder.decode_z2_to_z1(z2)

        z1 = pz1z2.sample()

        logits = self.decoder.decode_z1_to_x(z1)

        probs = tf.nn.sigmoid(logits)

        pxz1 = tfd.Bernoulli(logits=logits)

        x_sample = pxz1.sample()

        return x_sample, probs

    def conditional_sample(self, z2):
        pz1z2 = self.decoder.decode_z2_to_z1(z2)

        # now keep the z2 sample constant and draw several z1 samples
        z1 = pz1z2.sample(10)

        logits = self.decoder.decode_z1_to_x(z1)

        probs = tf.nn.sigmoid(logits)

        pxz1 = tfd.Bernoulli(logits=logits)

        x_sample = pxz1.sample()

        return x_sample, probs

    def generate_and_save_images(self, z2, epoch, string):

        # ---- samples from the prior
        x_samples, x_probs = self.sample(z2)
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

        # ---- only samples
        canvas = np.zeros((n * 28, n * 28))

        for i in range(n):
            for j in range(n):
                canvas[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = x_samples[i * n + j].reshape(28, 28)

        plt.clf()
        plt.figure(figsize=(10, 10))
        plt.imshow(canvas, cmap='gray_r')
        plt.title("epoch {:04d}".format(epoch), fontsize=50)
        plt.axis('off')
        plt.savefig('./results/' + string + '_samples_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

        # ---- conditional sampling, keeping z2 fixed, sampling several z1
        x_samples, x_probs = self.conditional_sample(z2)
        x_samples = x_samples.numpy().squeeze()

        n = 10

        canvas = np.zeros((n * 28, n * 28))

        for i in range(n):
            for j in range(n):
                canvas[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = x_samples[j, i].reshape(28, 28)

        plt.clf()
        plt.figure(figsize=(10, 10))
        plt.imshow(canvas, cmap='gray_r')
        plt.title("epoch {:04d}".format(epoch), fontsize=50)
        plt.axis('off')
        plt.savefig('./results/' + string + '_conditional_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

    def generate_samples(self,z):
        x_samples, x_probs = self.sample(z)
        return x_samples

    def generate_and_save_posteriors(self, x, y, n_samples, epoch, string):

        # ---- posterior snis means
        res = self.call(x, n_samples)

        snis_z1 = res["snis_z1"]
        snis_z2 = res["snis_z2"]

        # pca
        scaler = StandardScaler()

        z1 = scaler.fit_transform(snis_z1)

        pca = PCA(n_components=2)
        pca.fit(z1)
        z1 = pca.transform(z1)

        plt.clf()
        for c in np.unique(y):
            plt.scatter(z1[y == c, 0], z1[y == c, 1], s=10, label=str(c))
        plt.legend(loc=(1.04,0))
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.title("epoch {:04d}".format(epoch), fontsize=50)
        plt.savefig('./results/' + string + '_posterior_z1_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

        pca = PCA(n_components=2)
        pca.fit(snis_z2)
        z = pca.transform(snis_z2)

        plt.clf()
        for c in np.unique(y):
            plt.scatter(z[y == c, 0], z[y == c, 1], s=10, label=str(c))
        plt.legend(loc=(1.04,0))
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.title("epoch {:04d}".format(epoch), fontsize=50)
        plt.savefig('./results/' + string + '_posterior_z2_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

    @staticmethod
    def write_to_tensorboard(res, step):
        tf.summary.scalar('Evaluation/vae_elbo', res["vae_elbo"], step=step)
        tf.summary.scalar('Evaluation/iwae_elbo', res["iwae_elbo"], step=step)
        tf.summary.scalar('Evaluation/lpxz1', res['lpxz1'].numpy().mean(), step=step)
        tf.summary.scalar('Evaluation/lpz1z2', res['lpz1z2'].numpy().mean(), step=step)
        tf.summary.scalar('Evaluation/lqz1x', res['lqz1x'].numpy().mean(), step=step)
        tf.summary.scalar('Evaluation/lqz2z1', res['lqz2z1'].numpy().mean(), step=step)
        tf.summary.scalar('Evaluation/lpz2', res['lpz2'].numpy().mean(), step=step)
