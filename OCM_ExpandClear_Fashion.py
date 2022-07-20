
from tensorflow_probability import distributions as tfd
from tensorflow import keras
import numpy as np
import os
import argparse
import datetime
import time
import sys
sys.path.insert(0, './src')
import utils
import iwae1
import iwae2
import DMix
from data_hand import *
from keras.utils import to_categorical
from Utils2 import *
from utils import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# TODO: control warm-up from commandline
parser = argparse.ArgumentParser()
parser.add_argument("--stochastic_layers", type=int, default=1, choices=[1, 2], help="number of stochastic layers in the model")
parser.add_argument("--n_samples", type=int, default=50, help="number of importance samples")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--epochs", type=int, default=-1,
                    help="numper of epochs, if set to -1 number of epochs "
                         "will be set based on the learning rate scheme from the paper")
parser.add_argument("--objective", type=str, default="vae_elbo", choices=["vae_elbo", "iwae_elbo", "iwae_eq14", "vae_elbo_kl"])
parser.add_argument("--gpu", type=str, default='2', help="Choose GPU")
args = parser.parse_args()
print(args)
import numpy.linalg as la

# ---- string describing the experiment, to use in tensorboard and plots
string = "main_{0}_{1}_{2}".format(args.objective, args.stochastic_layers, args.n_samples)

'''
# ---- set the visible GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ---- dynamic GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
'''
# ---- number of passes over the data, see bottom of page 6 in [1]

# ---- load data
(Xtrain, ytrain), (Xtest, ytest) = keras.datasets.fashion_mnist.load_data()
Ntrain = Xtrain.shape[0]
Ntest = Xtest.shape[0]

# ---- reshape to vectors
Xtrain = Xtrain.reshape(Ntrain, -1) / 255
Xtest = Xtest.reshape(Ntest, -1) / 255

Xtest = utils.bernoullisample(Xtest)

# ---- do the training
start = time.time()
best = float(-np.inf)

#Split MNIST into Five tasks
y_train = to_categorical(ytrain, num_classes=10)
ytest = to_categorical(ytest, num_classes=10)
arr1, labelArr1, arr2, labelArr2, arr3, labelArr3, arr4, labelArr4, arr5, labelArr5,arr6, labelArr6,arr7, labelArr7,arr8, labelArr8,arr9, labelArr9,arr10, labelArr10 = Split_dataset_by10(Xtrain,y_train)

arr1_test, labelArr1_test, arr2_test, labelArr2_test, arr3_test, labelArr3_test, arr4_test, labelArr4, arr5_test, labelArr5_test,arr6_test, labelArr6_test,arr7_test, labelArr7_test,arr8_test, labelArr8_test,arr9_test, labelArr9_test,arr10_test, labelArr10_test = Split_dataset_by10(
    Xtest,
    ytest)

arr1_test = utils.bernoullisample(arr1_test)
arr2_test = utils.bernoullisample(arr2_test)
arr3_test = utils.bernoullisample(arr3_test)
arr4_test = utils.bernoullisample(arr4_test)
arr5_test = utils.bernoullisample(arr5_test)
arr6_test = utils.bernoullisample(arr6_test)
arr7_test = utils.bernoullisample(arr7_test)
arr8_test = utils.bernoullisample(arr8_test)
arr9_test = utils.bernoullisample(arr9_test)
arr10_test = utils.bernoullisample(arr10_test)

totalSet = np.concatenate((arr1,arr2,arr3,arr4,arr5,arr6,arr7,arr8,arr9,arr10),
                               axis=0)

testingSet = np.concatenate((arr1_test,arr2_test,arr3_test,arr4_test,arr5_test,arr6_test,arr7_test,arr8_test,arr9_test,arr10_test),
                               axis=0)


print(np.shape(totalSet))

taskCount = 5

totalResults = []

class LifeLone_MNIST(object):
    def __init__(self):
        self.batch_size = 64
        self.input_height = 28
        self.input_width = 28
        self.c_dim = 1
        self.z_dim = 50
        self.len_discrete_code = 4
        self.epoch = 50

        self.learning_rate = 0.0001
        self.beta1 = 0.5

        self.beta = 0.1
        self.data_dim = 28*28

        self.input_x = tf.placeholder(tf.float32, [self.batch_size, self.data_dim])
        self.input_test = tf.placeholder(tf.float32, [1, self.data_dim])

        self.text_k = tf.tile(self.input_x,[5000, 1])

        self.NofImportanceSamples = 50

        self.latentZArr = []
        self.latentXArr = []

        self.testLatentZArr = []
        self.testLatentXArr = []
        self.evalLatentZArr = []
        self.evalLatentXArr = []

        self.lossArr = []
        self.allLossArr = []
        self.recoArr = []
        self.testLossArr = []
        self.KlArr = []
        self.TestKLArr = []
        self.EvalKLArr = []
        self.EvaluationLossArr = []

        self.totalMemory = []
        self.maxCountForEach = self.batch_size * 2
        self.RecoOneArr = []

        self.LatestEvaluationLoss = 0

    def shoaared_encoder(self,name, x, z_dim=20, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(x, 200, activation=tf.nn.tanh)
        return l1

    def encoder(self,name, x, z_dim=50, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(x, 200, activation=tf.nn.tanh)
            #l2 = tf.layers.dense(l1, 200, activation=tf.nn.relu)
            mu = tf.layers.dense(l1, z_dim, activation=None)
            sigma = tf.layers.dense(l1, z_dim, activation=tf.exp)
            return mu, sigma

    def shared_decoder(self,name,z, z_dim=50, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(z, 200, activation=tf.nn.tanh)
            return l1

    def decoder(self,name,z, z_dim=50, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(z, 200, activation=tf.nn.relu)
            #l2 = tf.layers.dense(l1, 200, activation=tf.nn.relu)
            x_hat = tf.layers.dense(
                l1, self.data_dim, activation=None)
            return x_hat

    def logmeanexp(self,log_w, axis):
        max = tf.reduce_max(log_w, axis=axis)
        return tf.math.log(tf.reduce_mean(tf.exp(log_w - max), axis=axis)) + max

    def Give_Features_Function2(self,test):
        count = np.shape(test)[0]
        totalSamples = utils.bernoullisample(test)

        featureArr = []
        for i in range(count):
            single = totalSamples[i]
            single = np.reshape(single,(1,-1))
            feature = self.sess.run(self.Give_Feature2, feed_dict={self.input_test: single})
            featureArr.append(feature)

        featureArr = np.array(featureArr)
        return featureArr

    def Create_Component(self,index):
        if np.shape(self.lossArr)[0] == 0:
            sharedEncoderName = "sharedEncoder"
            encoderName = "Encoder" + str(index)
            sharedDecoderName = "sharedDecoder"
            decoderName = "Decoder" + str(index)
            x_k = self.input_x
            testX = self.input_test

            z_shared = self.shoaared_encoder(sharedEncoderName, x_k,self.z_dim, reuse=False)
            q_mu, q_std = self.encoder(encoderName, z_shared, self.z_dim, reuse=False)

            z_shared_2 = self.shoaared_encoder(sharedEncoderName, testX,self.z_dim, reuse=True)
            q_mu_2, q_std_2 = self.encoder(encoderName, z_shared_2, self.z_dim, reuse=True)
            self.Give_Feature2 = q_mu_2[0]

            n_samples = self.NofImportanceSamples
            qzx = tfd.Normal(q_mu, q_std + 1e-6)
            z = qzx.sample(n_samples)

            self.latentZArr.append(z)

            x_shared = self.shared_decoder(sharedDecoderName, z, self.z_dim, reuse=False)

            self.latentXArr.append(x_shared)

            logits = self.decoder(decoderName, x_shared, self.z_dim, reuse=False)

            pxz = tfd.Bernoulli(logits=logits)

            pz = tfd.Normal(0, 1)

            lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

            lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

            lpxz = tf.reduce_sum(pxz.log_prob(self.input_x), axis=-1)

            beta = 1.0
            log_w = lpxz + beta * (lpz - lqzx)

            self.allLossArr.append(tf.reduce_mean(log_w,axis=0))

            kl = (lpz - lqzx)
            self.KlArr.append(kl)

            # mean over samples and batch
            vae_elbo = tf.reduce_mean(tf.reduce_mean(log_w, axis=0), axis=-1)
            vae_elbo_kl = tf.reduce_mean(lpxz) - beta * tf.reduce_mean(kl)

            # ---- IWAE elbos
            # eq (8): logmeanexp over samples and mean over batch
            iwae_elbo = tf.reduce_mean(self.logmeanexp(log_w, axis=0), axis=-1)
            trainingloss = -vae_elbo

            self.lossArr.append(trainingloss)
            self.vaeLoss = trainingloss

            #testing loss
            n_samples = 1000

            #set 5000 if gpu has more memories
            #n_samples = 1000

            z = qzx.sample(n_samples)
            self.testLatentZArr.append(z)

            x_shared = self.shared_decoder(sharedDecoderName, z, self.z_dim, reuse=True)
            self.testLatentXArr.append(x_shared)

            z_ = qzx.sample(1)
            x_shared_ = self.shared_decoder(sharedDecoderName, z_, self.z_dim, reuse=True)
            self.Give_Feature = x_shared_
            self.Give_Feature = tf.reshape(self.Give_Feature,(self.batch_size,-1))
            logits_reco = self.decoder(decoderName, x_shared_, self.z_dim, reuse=True)
            pxz_reco = tfd.Bernoulli(logits=logits_reco)
            reco = pxz_reco.sample(1)
            self.reco = tf.reshape(reco,(-1,28,28,1))

            logits = self.decoder(decoderName, x_shared, self.z_dim, reuse=True)

            pxz = tfd.Bernoulli(logits=logits)

            pz = tfd.Normal(0, 1)

            lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

            lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

            lpxz = tf.reduce_sum(pxz.log_prob(self.input_x), axis=-1)

            kl = (lpz - lqzx)
            self.TestKLArr.append(kl)

            beta = 1.0
            log_w = lpxz + beta * (lpz - lqzx)
            test_iwae_elbo = tf.reduce_mean(self.logmeanexp(log_w, axis=0), axis=-1)

            self.testLossArr.append(test_iwae_elbo)
            #end of the test loss

            #begin of evaluation loss
            z_shared = self.shoaared_encoder(sharedEncoderName, testX, self.z_dim, reuse=True)

            q_mu, q_std = self.encoder(encoderName, z_shared, self.z_dim, reuse=True)
            qzx = tfd.Normal(q_mu, q_std + 1e-6)

            z = qzx.sample(1000)
            self.evalLatentZArr.append(z)

            x_shared = self.shared_decoder(sharedDecoderName, z, self.z_dim, reuse=True)
            self.evalLatentXArr.append(x_shared)

            logits = self.decoder(decoderName, x_shared, self.z_dim, reuse=True)
            pxz = tfd.Bernoulli(logits=logits)

            z_reco = qzx.sample(1)
            x_shared_reco = self.shared_decoder(sharedDecoderName, z_reco, self.z_dim, reuse=True)
            logits_reco = self.decoder(decoderName, x_shared_reco, self.z_dim, reuse=True)
            pxz_reco = tfd.Bernoulli(logits=logits_reco)
            reco = pxz_reco.sample(1)
            reco = tf.reshape(reco, (-1, 28, 28, 1))
            self.RecoOneArr.append(reco[0])

            pz = tfd.Normal(0, 1)

            lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

            lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

            kl = (lpz - lqzx)
            self.EvalKLArr.append(kl)

            lpxz = tf.reduce_sum(pxz.log_prob(testX), axis=-1)
            log_w = lpxz + beta * (lpz - lqzx)
            test_iwae_elbo = tf.reduce_mean(self.logmeanexp(log_w, axis=0), axis=-1)

            self.EvaluationLossArr.append(test_iwae_elbo)

            T_vars = tf.trainable_variables()
            with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
                self.vae_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                    .minimize(trainingloss, var_list=T_vars)

        else:
            sharedEncoderName = "sharedEncoder"
            encoderName = "Encoder" + str(index)
            sharedDecoderName = "sharedDecoder"
            decoderName = "Decoder" + str(index)

            testX = self.input_test
            x_k = self.input_x
            z_shared = self.shoaared_encoder(sharedEncoderName, x_k, self.z_dim, reuse=True)
            q_mu, q_std = self.encoder(encoderName, z_shared, self.z_dim, reuse=False)

            z_shared_2 = self.shoaared_encoder(sharedEncoderName, testX, self.z_dim, reuse=True)
            q_mu_2, q_std_2 = self.encoder(encoderName, z_shared_2, self.z_dim, reuse=True)
            self.Give_Feature2 = tf.concat((self.Give_Feature2,q_mu_2[0]),axis=0)

            n_samples = self.NofImportanceSamples
            qzx = tfd.Normal(q_mu, q_std + 1e-6)
            z = qzx.sample(n_samples)

            sumZ = z

            self.latentZArr.append(sumZ)
            latentX1 = self.shared_decoder(sharedDecoderName, sumZ, z_dim=50, reuse=True)

            z_ = qzx.sample(1)
            x_shared_ = self.shared_decoder(sharedDecoderName, z_, self.z_dim, reuse=True)
            self.Give_Feature = x_shared_
            self.Give_Feature = tf.reshape(self.Give_Feature, (self.batch_size, -1))

            sumZ_genertor = latentX1
            self.latentXArr.append(sumZ_genertor)

            logits = self.decoder(decoderName, sumZ_genertor, z_dim=50, reuse=False)

            pxz = tfd.Bernoulli(logits=logits)

            pz = tfd.Normal(0, 1)

            lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

            lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

            lpxz = tf.reduce_sum(pxz.log_prob(self.input_x), axis=-1)

            beta = 1.0
            kl = (lpz - lqzx)
            KLsum = kl

            self.KlArr.append(KLsum)

            log_w = lpxz + beta * KLsum

            self.allLossArr.append(tf.reduce_mean(log_w,axis=0))

            vae_elbo = tf.reduce_mean(tf.reduce_mean(log_w, axis=0), axis=-1)
            iwae_elbo = tf.reduce_mean(self.logmeanexp(log_w, axis=0), axis=-1)
            trainingloss = -vae_elbo
            self.lossArr.append(trainingloss)

            #testing loss
            qzx_test = tfd.Normal(q_mu, q_std + 1e-6)
            n_samples = 1000
            #set 5000 if gpu has more memories
            #n_samples = 1000
            z_test = qzx.sample(n_samples)

            sumZ_test = z_test

            self.testLatentZArr.append(sumZ_test)

            latentX1_test = self.shared_decoder(sharedDecoderName, sumZ_test, z_dim=50, reuse=True)

            sumZ_genertor_test = latentX1_test

            self.testLatentXArr.append(sumZ_genertor_test)
            logits_test = self.decoder(decoderName, sumZ_genertor_test, z_dim=50, reuse=True)

            pxz_test = tfd.Bernoulli(logits=logits_test)

            pz_test = tfd.Normal(0, 1)

            lpz_test = tf.reduce_sum(pz_test.log_prob(z_test), axis=-1)

            lqzx_test = tf.reduce_sum(qzx_test.log_prob(z_test), axis=-1)

            lpxz_test = tf.reduce_sum(pxz_test.log_prob(self.input_x), axis=-1)

            beta = 1.0
            kl = (lpz_test - lqzx_test)
            KLsum = kl

            self.TestKLArr.append(KLsum)

            log_w_test = lpxz_test + beta * KLsum
            iwae_elbo_test = tf.reduce_mean(self.logmeanexp(log_w_test, axis=0), axis=-1)
            self.testLossArr.append(iwae_elbo_test)
            #end of testing loss

            #begin of evaluation loss
            z_shared = self.shoaared_encoder(sharedEncoderName, testX, self.z_dim, reuse=True)

            q_mu, q_std = self.encoder(encoderName, z_shared, self.z_dim, reuse=True)

            qzx_test = tfd.Normal(q_mu, q_std + 1e-6)
            n_samples = 5000
            # set 5000 if gpu has more memories
            # n_samples = 1000
            z_test = qzx_test.sample(1000)

            sumZ_test = z_test

            self.evalLatentZArr.append(sumZ_test)

            latentX1_test = self.shared_decoder(sharedDecoderName, sumZ_test, z_dim=50, reuse=True)

            sumZ_genertor_test = latentX1_test

            self.evalLatentXArr.append(sumZ_genertor_test)
            logits_test = self.decoder(decoderName, sumZ_genertor_test, z_dim=50, reuse=True)

            z_reco = qzx_test.sample(1)
            x_shared_reco = self.shared_decoder(sharedDecoderName, z_reco, self.z_dim, reuse=True)
            logits_reco = self.decoder(decoderName, x_shared_reco, self.z_dim, reuse=True)
            pxz_reco = tfd.Bernoulli(logits=logits_reco)
            reco = pxz_reco.sample(1)
            reco = tf.reshape(reco, (-1, 28, 28, 1))
            self.RecoOneArr.append(reco[0])

            pxz_test = tfd.Bernoulli(logits=logits_test)

            pz_test = tfd.Normal(0, 1)

            lpz_test = tf.reduce_sum(pz_test.log_prob(z_test), axis=-1)

            lqzx_test = tf.reduce_sum(qzx_test.log_prob(z_test), axis=-1)

            lpxz_test = tf.reduce_sum(pxz_test.log_prob(testX), axis=-1)

            beta = 1.0
            kl = (lpz_test - lqzx_test)
            KLsum = kl

            self.EvalKLArr.append(KLsum)

            log_w_test = lpxz_test + beta * KLsum
            iwae_elbo_test = tf.reduce_mean(self.logmeanexp(log_w_test, axis=0), axis=-1)
            self.EvaluationLossArr.append(iwae_elbo_test)

            T_vars = tf.trainable_variables()
            vars1 = [var for var in T_vars if var.name.startswith(decoderName)]
            vars2 = [var for var in T_vars if var.name.startswith(encoderName)]
            vars3 = [var for var in T_vars if var.name.startswith(sharedEncoderName)]
            vars4 = [var for var in T_vars if var.name.startswith(sharedDecoderName)]

            vars = vars1 + vars2# + vars3 + vars4

            self.vaeLoss = trainingloss
            with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
                self.vae_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                    .minimize(trainingloss, var_list=vars)

            global_vars = tf.global_variables()
            is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
            self.sess.run(tf.variables_initializer(not_initialized_vars))

    def Select_Component(self,single):

        lossArr = []
        for j in range(np.shape(self.EvaluationLossArr)[0]):
            loss = self.sess.run(self.EvaluationLossArr[j], feed_dict={self.input_test: single})
            lossArr.append(loss)

        minIndex = np.argmax(lossArr)
        return minIndex

    def Evaluation(self,test,index):
        mycount = int(np.shape(test)[0] / self.batch_size)
        sumLoss = 0
        for i in range(mycount):
            batch = test[i * self.batch_size: (i + 1) * self.batch_size]
            loss = self.sess.run(self.testLossArr[index], feed_dict={self.input_x: batch})
            sumLoss = sumLoss + loss
        sumLoss = sumLoss / mycount
        return sumLoss

    def GiveReconstruction(self,test):
        mycount = np.shape(test)[0]
        arr = []
        for i in range(mycount):
            single = test[i]
            single = np.reshape(single,(1,28*28))
            index = self.Select_Component(single)
            reco = self.sess.run(self.RecoOneArr[index], feed_dict={self.input_test: single})
            arr.append(reco)
        arr = np.array(arr)
        return arr

    def EvaluationAndIndex(self,test):
        mycount = np.shape(test)[0]
        sumLoss = 0
        for i in range(mycount):
            single = test[i]
            single = np.reshape(single,(1,28*28))
            index = self.Select_Component(single)
            loss = self.sess.run(self.EvaluationLossArr[index], feed_dict={self.input_test: single})
            sumLoss = sumLoss + loss

        sumLoss = sumLoss / mycount
        return sumLoss,index

    def Build(self):
        #Build the first component

        self.Create_Component(1)
        #self.Create_Component(2)

    def Give_Features_Function(self,test):
        count = np.shape(test)[0]
        newCount = int(count / self.batch_size)
        remainCount = count - newCount * self.batch_size
        remainSamples = test[newCount * self.batch_size:count]
        remainSamples = np.concatenate((remainSamples, test[0:(self.batch_size - remainCount)]), axis=0)
        remainSamples = utils.bernoullisample(remainSamples)
        totalSamples = utils.bernoullisample(test)

        featureArr = []
        for i in range(newCount):
            batch = totalSamples[i * self.batch_size:(i + 1) * self.batch_size]
            features = self.sess.run(self.Give_Feature, feed_dict={self.input_x: batch})
            for j in range(self.batch_size):
                featureArr.append(features[j])

        ff = self.sess.run(self.Give_Feature, feed_dict={self.input_x: remainSamples})
        for i in range(remainCount):
            featureArr.append(ff[i])

        featureArr = np.array(featureArr)
        return featureArr

    def gaussian(self,sigma,x,y):
        return np.exp(-np.sqrt(la.norm(x - y) ** 2 / (2 * sigma ** 2)))

    def SelectSample_InMemory(self):
        sigma = 10

        dynamicFeatureArr = self.Give_Features_Function2(self.DynamicMmeory)
        fixedFeatureArr = self.Give_Features_Function2(self.FixedMemory)

        count = np.shape(dynamicFeatureArr)[0]
        count2 = np.shape(fixedFeatureArr)[0]
        relationshipMatrix = np.zeros((count, count2))
        for i in range(count):
            for j in range(count2):
                relationshipMatrix[i, j] = self.gaussian(sigma, dynamicFeatureArr[i], fixedFeatureArr[j])

        sampleDistance = []
        for i in range(count):
            sum1 = 0
            for j in range(count2):
                sum1 = sum1 + relationshipMatrix[i, j]
            sum1 = sum1 / count2
            sampleDistance.append(sum1)

        sampleDistance = np.array(sampleDistance)
        totalSum = np.mean(sampleDistance)

        index = np.argsort(-sampleDistance)
        self.DynamicMmeory = self.DynamicMmeory[index]
        sampleDistance = sampleDistance[index]

        print(sampleDistance)

        #Evaluation of building a new component
        memory = np.concatenate((self.DynamicMmeory,self.FixedMemory),axis=0)
        memory = np.array(memory)
        sumLoss = 0
        sumEvaluation = 0
        for j in range(np.shape(self.EvaluationLossArr)[0]):
            sumEvaluation = sumEvaluation + self.EvaluationLossArr[j]
        sumEvaluation = sumEvaluation / int(np.shape(self.EvaluationLossArr)[0])

        memoryCount = np.shape(memory)[0]
        for i in range(memoryCount):
            single = memory[i]
            single = np.reshape(single,(1,-1))
            loss = self.sess.run(sumEvaluation,feed_dict={self.input_test:single})
            sumLoss = sumLoss + loss
        sumLoss = sumLoss / memoryCount

        if self.LatestEvaluationLoss == 0:
            self.LatestEvaluationLoss = sumLoss
        else:
            diff = np.abs(self.LatestEvaluationLoss - sumLoss)
            if diff > 10:
                if np.shape(self.lossArr)[0] < 10:
                    self.Create_Component(np.shape(self.lossArr)[0] + 1)
                    self.FixedMemory = []
            self.LatestEvaluationLoss = sumLoss
            print("total")
            print(diff)

        if np.shape(self.FixedMemory)[0] < self.maxMmeorySize * 5:
            print("diff")
            for i in range(count):
                if i > 13:
                    break

                if sampleDistance[i] > self.ThresholdForFixed:
                    single = self.DynamicMmeory[i]
                    single = np.reshape(single,(1,-1))

                    if np.shape(self.FixedMemory)[0] == 0:
                        self.FixedMemory = single
                    else:
                        self.FixedMemory = np.concatenate((self.FixedMemory,single),axis=0)

                    print(sampleDistance[i])
                else:
                    break

    def Train(self):
        pz = tfd.Normal(0, 1)
        step = 0
        taskCount = 1

        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        config.gpu_options.allow_growth = True

        self.totalSet = totalSet
        self.totalSet = np.array(self.totalSet)
        self.totalSet = np.reshape(self.totalSet,(-1,28*28))

        self.FixedMemory = self.totalSet[0:self.batch_size]
        self.ThresholdForFixed = 1

        self.minThreshold = 0.0005
        self.maxThreshold = 0.05

        self.DynamicMmeory =self.totalSet[0:self.batch_size]
        self.maxMmeorySize = 512

        self.DynamicMmeory = np.array(self.DynamicMmeory)
        totalCount = int(np.shape(self.totalSet)[0] / self.batch_size)

        self.moveThreshold = (self.maxThreshold - self.minThreshold) / totalCount

        sourceRiskArr = []
        targetRiskArr = []

        with tf.Session(config=config) as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            for index in range(totalCount):
                batch = self.totalSet[index * self.batch_size : (index + 1) * self.batch_size]
                epochs = 100
                currentX = Xtrain

                if np.shape(self.DynamicMmeory)[0] == 0:
                    self.DynamicMmeory = batch
                else:
                    self.DynamicMmeory = np.concatenate((self.DynamicMmeory, batch), axis=0)

                if np.shape(self.FixedMemory)[0] == 0:
                    self.FixedMemory = batch

                #self.Create_Component(2)
                for epoch in range(epochs):
                    # ---- binarize the training data at the start of each epoch
                    batch_binarized = utils.bernoullisample(self.FixedMemory)
                    Xtrain_binarized = utils.bernoullisample(self.DynamicMmeory)
                    Xtrain_binarized = np.concatenate((Xtrain_binarized,batch_binarized),axis=0)

                    n_examples = np.shape(Xtrain_binarized)[0]
                    index2 = [i for i in range(n_examples)]
                    np.random.shuffle(index2)
                    Xtrain_binarized = Xtrain_binarized[index2]
                    counter = 0

                    myCount = int(np.shape(Xtrain_binarized)[0] / self.batch_size)

                    for idx in range(myCount):
                        step = step + 1
                        step = step %100000

                        batchImages = Xtrain_binarized[idx*self.batch_size:(idx+1)*self.batch_size]
                        beta = 1.0
                        _, d_loss = self.sess.run([self.vae_optim, self.vaeLoss],
                                                  feed_dict={self.input_x: batchImages})

                    if epoch % 2 == 0:
                        print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
                                  .format(epoch, epochs, index, totalCount, d_loss, np.shape(self.lossArr)[0], 0))

                # Evaluate the novelty of a new batch of samples
                if np.shape(self.DynamicMmeory)[0] > self.maxMmeorySize:
                    self.SelectSample_InMemory()
                    self.ThresholdForFixed = 0.35#self.maxThreshold - index * self.moveThreshold
                    self.DynamicMmeory = []

            print("Memory size")
            print(np.shape(self.FixedMemory)[0])

            print("Number of components")
            print(np.shape(self.lossArr)[0])

            test1, index1 = self.EvaluationAndIndex(Xtest)
            batch = Xtest[0:self.batch_size]
            x_batch = np.reshape(batch,(-1,28,28,1))
            reco = self.GiveReconstruction(x_batch)
            reco = reco * 255.0
            x_batch = x_batch * 255.0
            cv2.imwrite(os.path.join("results/", 'OnlineVAEExpandClear_Fashion_real.png'), merge2(x_batch[:64], [8, 8]))
            cv2.imwrite(os.path.join("results/", 'OnlineVAEExpandClear_Fashion_reco.png'), merge2(reco[:64], [8, 8]))

            print(test1)

model = LifeLone_MNIST()
model.Build()
model.Train()


'''
# ---- save final weights
model.save_weights('/tmp/iwae/{0}/final_weights'.format(string))

# ---- load the final weights?
# model.load_weights('/tmp/iwae/{0}/final_weights'.format(string))

# ---- test-set llh estimate using 5000 samples
test_elbo_metric = utils.MyMetric()
L = 5000

# ---- since we are using 5000 importance samples we have to loop over each element of the test-set


for i, x in enumerate(Xtest):
    res = model(x[None, :].astype(np.float32), L)
    test_elbo_metric.update_state(res['iwae_elbo'][None, None])
    if i % 200 == 0:
        print("{0}/{1}".format(i, Ntest))

test_set_llh = test_elbo_metric.result()
test_elbo_metric.reset_states()

print("Test-set {0} sample log likelihood estimate: {1:.4f}".format(L, test_set_llh))
'''