import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # needed when running from commandline
import matplotlib.pyplot as plt
import cycler
import utils
import tensorflow.compat.v1 as tf1


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


class VAENode(tf.keras.Model):
    def __init__(self,
                 n_hidden,isBasic,
                 **kwargs):
        super(VAENode, self).__init__(**kwargs)

        self.objFun = 0
        n_latent = 100
        n_hidden = 200
        self.logLikelihood = 0
        self.IsBasic = isBasic

        self.CurrentModel = 0

        #encoder layers
        if isBasic == True:
            self.SharedEncoder = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.specificEncoder_net = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.specificEncoder_mu = tf.keras.layers.Dense(n_latent, activation=None)
        self.specificEncoder_std = tf.keras.layers.Dense(n_latent, activation=tf.exp)

        #decoder layers
        if isBasic == True:
            self.SharedDecoder = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.SpecificDecoder_layer1 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.SpecificDecoder_output = tf.keras.layers.Dense(784, activation=None,bias_initializer=utils.get_bias())

        self.ComponentWeights = []
        self.BasicNodes = []
        self.basicCount = 0

        if isBasic == True:
            self.ModelParameters = self.SharedEncoder.trainable_variables+self.specificEncoder_net.trainable_variables+self.specificEncoder_mu.trainable_variables+self.specificEncoder_std.trainable_variables+ self.SharedDecoder.trainable_variables+self.SpecificDecoder_layer1.trainable_variables+self.SpecificDecoder_output.trainable_variables
        else:
            self.ModelParameters = self.specificEncoder_net.trainable_variables + self.specificEncoder_mu.trainable_variables + self.specificEncoder_std.trainable_variables + self.SpecificDecoder_layer1.trainable_variables + self.SpecificDecoder_output.trainable_variables

    def SetTrainable(self,isTrainable):
        if self.IsBasic == True:
            self.SharedEncoder.trainable = isTrainable
        self.specificEncoder_net.trainable = isTrainable
        self.specificEncoder_mu.trainable = isTrainable
        self.specificEncoder_std.trainable = isTrainable

        #decoder layers
        if self.IsBasic == True:
            self.SharedDecoder.trainable = isTrainable
        self.SpecificDecoder_layer1.trainable = isTrainable
        self.SpecificDecoder_output.trainable = isTrainable


    def call(self, x, n_samples, beta=1.0):
        bound = 0
        if self.IsBasic == True:
            output = self.SharedEncoder(x)
            output = self.specificEncoder_net(output)
            q_mu = self.specificEncoder_mu(output)
            q_std = self.specificEncoder_std(output)

            qzx = tfd.Normal(q_mu, q_std + 1e-6)
            z = qzx.sample(n_samples)

            decoderOutput = self.SharedDecoder(z)
            decoderOutput = self.SpecificDecoder_layer1(decoderOutput)
            logits = self.SpecificDecoder_output(decoderOutput)

            pxz = tfd.Bernoulli(logits=logits)

            pz = tfd.Normal(0, 1)

            lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

            lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

            lpxz = tf.reduce_sum(pxz.log_prob(x), axis=-1)

            log_w = lpxz + beta * (lpz - lqzx)
            iwae_elbo = tf.reduce_mean(utils.logmeanexp(log_w, axis=0), axis=-1)
            bound = iwae_elbo
        else:
            z_sum = 0
            kl_sum = 0
            pz = tfd.Normal(0, 1)

            basicCount = self.basicCount
            # encoding
            for i in range(basicCount):
                basicNode = self.BasicNodes[i]
                output = basicNode.SharedEncoder(x)
                output = basicNode.specificEncoder_net(output)
                q_mu = basicNode.specificEncoder_mu(output)
                q_std = basicNode.specificEncoder_std(output)
                qzx = tfd.Normal(q_mu, q_std + 1e-6)
                z = qzx.sample(n_samples)
                z_sum = z_sum + z * self.ComponentWeights[i]

                lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)
                lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)
                kl = (lpz - lqzx)
                kl_sum = kl_sum + self.ComponentWeights[i] * kl

            # decoding
            sumX = 0
            for i in range(basicCount):
                basicNode = self.BasicNodes[i]
                decoderOutput = basicNode.SharedDecoder(z_sum)
                sumX = sumX + decoderOutput * self.ComponentWeights[i]

            reco = self.SpecificDecoder_layer1(sumX)
            logits = self.SpecificDecoder_output(reco)

            beta = 1.0
            pxz = tfd.Bernoulli(logits=logits)
            lpxz = tf.reduce_sum(pxz.log_prob(x), axis=-1)
            log_w = lpxz + beta * kl_sum
            iwae_elbo = tf.reduce_mean(utils.logmeanexp(log_w, axis=0), axis=-1)
            bound = iwae_elbo

        return bound

    def Build_NormalNode(self,x,basicNodes,basicCount,weights,n_samples):

        z_sum = 0
        #encoding
        z_sum = 0
        kl_sum = 0
        pz = tfd.Normal(0, 1)

        #encoding
        for i in range(basicCount):
            basicNode = basicNodes[i]
            output = basicNode.SharedEncoder(x)
            output = basicNode.specificEncoder_net(output)
            q_mu = basicNode.specificEncoder_mu(output)
            q_std = basicNode.specificEncoder_std(output)
            qzx = tfd.Normal(q_mu, q_std + 1e-6)
            z = qzx.sample(n_samples)
            z_sum = z_sum + z*weights[i]

            lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)
            lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)
            kl = (lpz - lqzx)
            kl_sum = kl_sum + weights[i]*kl

        #decoding
        sumX = 0
        for i in range(basicCount):
            basicNode = basicNodes[i]
            decoderOutput = basicNode.SharedDecoder(z_sum)
            sumX = sumX + decoderOutput*weights[i]

        reco = self.SpecificDecoder_layer1(sumX)
        logits = self.SpecificDecoder_output(reco)

        beta = 1.0
        pxz = tfd.Bernoulli(logits=logits)
        lpxz = tf.reduce_sum(pxz.log_prob(x), axis=-1)
        log_w = lpxz + beta * kl_sum
        vae_elbo = tf.reduce_mean(tf.reduce_mean(log_w, axis=0), axis=-1)
        self.objFun = -vae_elbo
        #decoding
        return vae_elbo

    def Build_BasicNode(self,x,n_samples):

        output = self.SharedEncoder(x)
        output = self.specificEncoder_net(output)
        q_mu = self.specificEncoder_mu(output)
        q_std = self.specificEncoder_std(output)

        qzx = tfd.Normal(q_mu, q_std + 1e-6)
        z = qzx.sample(n_samples)

        decoderOutput = self.SharedDecoder(z)
        decoderOutput = self.SpecificDecoder_layer1(decoderOutput)
        logits = self.SpecificDecoder_output(decoderOutput)

        pxz = tfd.Bernoulli(logits=logits)

        pz = tfd.Normal(0, 1)

        lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

        lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

        lpxz = tf.reduce_sum(pxz.log_prob(x), axis=-1)

        beta = 1.0
        log_w = lpxz + beta * (lpz - lqzx)
        vae_elbo = tf.reduce_mean(tf.reduce_mean(log_w, axis=0), axis=-1)
        self.objFun = -vae_elbo
        return vae_elbo

class DMix(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(DMix, self).__init__(**kwargs)

        self.currentTaskIndex = 0
        self.basic_number = 1
        self.BasicNodeArr = []
        self.AllNodeArr = []
        self.currentNode = 0
        self.n_samples = 50
        self.threshold = 100
        self.batch_size = 20
        self.Componentweights = []
        self.basicIndexArr = []
        self.currentIndex = 0

        self.ShownWeights = np.zeros((6,6))

        #self.encoder = Encoder(n_hidden, n_latent)
        #self.decoder = Decoder(n_hidden)

    @tf.function
    def train_step(self, x, n_samples, beta, optimizer, objective="vae_elbo"):
        with tf.GradientTape() as tape:
            node = self.AllNodeArr[self.currentIndex]
            print(self.currentIndex)
            if node.IsBasic == True:
                res = self.currentNode.Build_BasicNode(x,n_samples)
            else:
                b = 0
                res = self.currentNode.Build_NormalNode(x,self.BasicNodeArr,self.basic_number,self.Componentweights,self.n_samples)

            loss = -res

        trainable_weights = self.trainable_weights
        grads = tape.gradient(loss, trainable_weights)
        optimizer.apply_gradients(zip(grads, trainable_weights))

        return res

    def Create_New_Component(self,x,data,isFirst):
        #The first task learning
        self.currentTaskIndex = self.currentTaskIndex+1
        newModel = 0
        if isFirst == True:
            newModel = VAENode(200,True)
            newModel.Build_BasicNode(x,self.n_samples)
            self.AllNodeArr.append(newModel)
            self.currentNode = newModel
            self.BasicNodeArr.append(newModel)
            self.basicIndexArr.append(self.currentTaskIndex-1)
        else:
            #Evaluate Similarity matrix
            arr = []
            weights = []
            sumWeight =0
            count = int(np.shape(data)[0]/self.batch_size)
            for i in range(self.basic_number):
                basicNode = self.BasicNodeArr[i]
                lossSum = 0
                for j in range(count):
                    x1 = data[j*self.batch_size:(j+1)*self.batch_size]
                    loss = basicNode(x1,self.n_samples)
                    lossSum = lossSum+loss
                lossSum = lossSum / count

                diff = np.abs(lossSum - basicNode.logLikelihood)
                arr.append(diff)
                sumWeight = sumWeight+diff

            isExpansion= False
            minum = np.min(arr)
            print("minum")
            print(minum)
            if minum > self.threshold:
                isExpansion = True

            if isExpansion == True:
                print("build basic")
                newModel = VAENode(200, True)
                newModel.Build_BasicNode(x, self.n_samples)
                self.AllNodeArr.append(newModel)
                self.currentNode = newModel
                self.BasicNodeArr.append(newModel)
                self.basic_number = self.basic_number + 1
                self.basicIndexArr.append(self.currentTaskIndex - 1)
            else:
                print("build a specific")
                newWeights = 0
                for i in range(self.basic_number):
                    newWeights = newWeights + (sumWeight - arr[i])

                for i in range(self.basic_number):
                    a = (sumWeight - arr[i]) /newWeights
                    weights.append(a)
                self.Componentweights = weights

                #showing weights in a matrix
                for hh1 in range(np.shape(weights)[0]):
                    index = self.basicIndexArr[hh1]
                    self.ShownWeights[self.currentTaskIndex-1,index] = weights[hh1]

                #calculate normal node
                newModel = VAENode(200, False)
                newModel.Build_NormalNode(x,self.BasicNodeArr,self.basic_number,weights,self.n_samples)
                newModel.ComponentWeights = self.Componentweights.copy()
                newModel.BasicNodes = self.BasicNodeArr
                newModel.basicCount = np.shape(self.BasicNodeArr)[0]
                self.AllNodeArr.append(newModel)
                self.currentNode = newModel

        return newModel
    def Calculate_NLL_By_SelectedComponent(self,component,n_samples,textX):
        node = self.AllNodeArr[component]
        count = int(np.shape(textX)[0]/self.batch_size)
        sumLoss = 0
        for i in range(count):
            xx = textX[i*self.batch_size:(i+1)*self.batch_size]
            loss = node(xx,n_samples)
            sumLoss = sumLoss+loss
        sumLoss = sumLoss / count
        sumLoss = np.abs(sumLoss)
        return sumLoss

    def SetTranable(self,isTraiable):
        for i in range(np.shape(self.AllNodeArr)[0]):
            node = self.AllNodeArr[i]
            node.SetTrainable(isTraiable)

    def Evaluation(self,textX,n_samples):
        arr = []
        for i in range(np.shape(self.AllNodeArr)[0]):
            loss = self.Calculate_NLL_By_SelectedComponent(i,n_samples,textX)
            arr.append(loss)
        arr = np.array(arr)
        print(arr)
        minvalue = np.min(arr)
        minIndex = np.argmin(arr)
        minIndex = minIndex+1
        return minIndex,minvalue

    def call(self, x, n_samples, beta=1.0):
        # ---- encode/decode
        z, qzx  = self.encoder(x, n_samples)

        logits, pxz = self.decoder(z)

        # ---- loss
        pz = tfd.Normal(0, 1)

        lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

        lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

        lpxz = tf.reduce_sum(pxz.log_prob(x), axis=-1)

        beta = 1.0
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




