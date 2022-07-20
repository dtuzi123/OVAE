

import numpy as np
from ops import *
from utils import *
from Utlis2 import *
import tf_slim as slim

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')
d_bn4 = batch_norm(name='d_bn4')
d_bn5 = batch_norm(name='d_bn5')

'''
e_bn2 = batch_norm(name='e_bn2')
e_bn3 = batch_norm(name='e_bn3')
e_bn4 = batch_norm(name='e_bn4')
'''
g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')
g_bn4 = batch_norm(name='g_bn4')
g_bn5 = batch_norm(name='g_bn5')
g_bn6 = batch_norm(name='g_bn6')
g_bn7 = batch_norm(name='g_bn7')

def Encoder_MNIST_Supervised(image,y,name, batch_size=64, reuse=False):
    with tf.compat.v1.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        image = tf.compat.v1.reshape(image,(-1,28*28))
        #image = tf.compat.v1.concat((image,y),axis=1)
        net = tf.compat.v1.nn.relu(bn(linear(image, 256, scope='g_fc1'),is_training=True, scope='g_bn1'))

        batch_size = 64
        kernel = 3
        z_dim = 256
        h5 = linear(image, 256, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 10
        logoutput = linear(h5, continous_len, 'e_log_sigma_sq')
        softmaxValue = tf.compat.v1.nn.softmax(logoutput)

        return logoutput, softmaxValue


def Encoder_MNIST_Supervised_3(image,y,name, batch_size=64, reuse=False):
    with tf.compat.v1.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        image = tf.compat.v1.reshape(image,(-1,28*28))
        #image = tf.compat.v1.concat((image,y),axis=1)
        net = tf.compat.v1.nn.relu(bn(linear(image, 256, scope='g_fc1'),is_training=True, scope='g_bn1'))

        batch_size = 64
        kernel = 3
        z_dim = 256
        h5 = linear(image, 256, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 10
        logoutput = linear(h5, continous_len, 'e_log_sigma_sq')
        softmaxValue = tf.compat.v1.nn.softmax(logoutput)

        return logoutput, softmaxValue,h5


def Generator_SharedMNIST(name, z, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # ArchitecDiscriminator_Celebature : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        # merge noise and code
        batch_size = 64
        net = tf.compat.v1.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))

        '''
        net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
        net = tf.reshape(net, [batch_size, 7, 7, 128])
        net = tf.nn.relu(
            bn(deconv2d(net, [batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
               scope='g_bn3'))

        out = tf.nn.sigmoid(deconv2d(net, [batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

        return out
        '''
        return net

def Generator_SubMNIST(name, z, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # ArchitecDiscriminator_Celebature : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        # merge noise and code
        batch_size = 64
        net = tf.compat.v1.nn.relu(bn(linear(z, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
        net = tf.compat.v1.reshape(net, [batch_size, 7, 7, 128])
        net = tf.compat.v1.nn.relu(
            bn(deconv2d(net, [batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
               scope='g_bn3'))

        out = tf.compat.v1.nn.sigmoid(deconv2d(net, [batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

    return out


def Give_TotalParameters():
    total_parameters = 0
    # iterating over all variables
    for variable in tf.all_variables():  # 统计全部的参数
        # for variable in tf.trainable_variables():#统计可以训练的参数
        local_parameters = 1
        shape = variable.get_shape()  # getting shape of a variable
        for i in shape:
            local_parameters *= i.value  # mutiplying dimension values
            total_parameters += local_parameters

    return total_parameters

def Input64_SuperEncoder(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            input = tf.reshape(input, (-1, 64 * 64 * 3))
            net = lrelu(bn(linear(input, 2048, scope='c_fc6'), is_training=is_training, scope='c_bn6'))
            net = lrelu(bn(linear(net, 1024, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
            net = lrelu(bn(linear(net, 500, scope='c_fc3'), is_training=is_training, scope='c_bn3'))
            #net = lrelu(bn(linear(net, 500, scope='c_fc4'), is_training=is_training, scope='c_bn4'))
            z_dim = 256
            z_mean = linear(net, z_dim, 'e_mean')
            z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
            z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq

def Input64_SubEncoder(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            z_dim = 256
            net = input
            net = lrelu(bn(linear(net, 500, scope='c_fc4'), is_training=is_training, scope='c_bn4'))

            z_mean = linear(net, z_dim, 'e_mean')
            z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
            z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq

def Input64_SuperGenerator(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            z = input
            kernel = 3
            myScale = 1
            # fully-connected layers
            net = lrelu(bn(linear(input, 512, scope='c_fc4'), is_training=is_training, scope='c_bn4'))
            net = lrelu(bn(linear(input, 512, scope='c_fc2'), is_training=is_training, scope='c_bn2'))
            net = lrelu(bn(linear(net, 1024, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
            net = lrelu(bn(linear(net, 2048, scope='c_fc3'), is_training=is_training, scope='c_bn3'))
            z_dim = 2048
            z_mean = linear(net, z_dim, 'e_mean')
            z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
            z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq

def Input64_SubGenerator(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):

            net = lrelu(bn(linear(input, 2048, scope='c_fc6'), is_training=is_training, scope='c_bn6'))
            net = lrelu(bn(linear(net, 3000, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
            output = linear(net, 64 * 64 * 3, scope='c_fc4')
            output = tf.nn.sigmoid(output)

        return output


def CIFAR_SuperEncoder(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        imageDim = 32*32*3
        dim1 = 600

        input = tf.reshape(input,(-1,32*32*3))

        w_init = tf.glorot_uniform_initializer()
        dim1 = 2000
        dim2 = 1500
        dim3 = 1000
        dim4 = 600
        dim5 = 300
        dim6 = 200

        w1 = tf.get_variable('w1', [imageDim, dim1], initializer=w_init)
        w2 = tf.get_variable('w2', [dim1, dim2], initializer=w_init)
        w3 = tf.get_variable('w3', [dim2, dim3], initializer=w_init)
        w3_1 = tf.get_variable('w3_1', [dim2, dim3], initializer=w_init)

        z1 = tf.matmul(input, w1)
        z1 = tf.matmul(z1, w2)
        #z1 = tf.matmul(z1, w3)

        z_mean = tf.matmul(z1, w3)
        z_variance = tf.nn.softplus(tf.matmul(z1, w3_1))

    return z_mean,z_variance

def CIFAR_SuperEncoder_2(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        imageDim = 32*32*3
        dim1 = 600

        input = tf.reshape(input,(-1,32*32*3))

        w_init = tf.glorot_uniform_initializer()
        dim1 = 2000
        dim2 = 1500
        dim3 = 1000
        dim4 = 600
        dim5 = 300
        dim6 = 200

        w1 = tf.get_variable('w1', [imageDim, dim1], initializer=w_init)
        w2 = tf.get_variable('w2', [dim1, dim2], initializer=w_init)
        w2_1 = tf.get_variable('w2_1', [dim1, dim2], initializer=w_init)

        #w3 = tf.get_variable('w3', [dim2, dim3], initializer=w_init)
        #w3_1 = tf.get_variable('w3_1', [dim2, dim3], initializer=w_init)

        z1 = tf.matmul(input, w1)
        #z1 = tf.matmul(z1, w2)
        #z1 = tf.matmul(z1, w3)

        z_mean = tf.matmul(z1, w2)
        z_variance = tf.nn.softplus(tf.matmul(z1, w2_1))

    return z_mean,z_variance

def CIFAR_SuperEncoder_Deep(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            input = tf.reshape(input, (-1, 32 * 32 * 3))
            net = lrelu(bn(linear(input, 2000, scope='c_fc6'), is_training=is_training, scope='c_bn6'))
            net = lrelu(bn(linear(net, 1500, scope='c_fc1'), is_training=is_training, scope='c_bn1'))

            z_dim = 1000
            z_mean = linear(net, z_dim, 'e_mean')

        return z_mean

def CIFAR_SubEncoder_Deep(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            z_dim = 200
            net = input
            net = lrelu(bn(linear(net, 1000, scope='c_fc4'), is_training=is_training, scope='c_bn4'))
            net = lrelu(bn(linear(net, 600, scope='c_fc5'), is_training=is_training, scope='c_bn5'))
            net = lrelu(bn(linear(net, 300, scope='c_fc7'), is_training=is_training, scope='c_bn7'))

            z_mean = linear(net, z_dim, 'e_mean')
            z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
            z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq

def CIFAR_SubEncoder(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        w_init = tf.glorot_uniform_initializer()
        dim1 = 2000
        dim2 = 1500
        dim3 = 1000
        dim4 = 600
        dim5 = 300
        dim6 = 200

        w4 = tf.get_variable('w4', [dim3, dim4], initializer=w_init)
        w5 = tf.get_variable('w5', [dim4, dim5], initializer=w_init)
        w6 = tf.get_variable('w6', [dim5, dim6], initializer=w_init)
        w6_1 = tf.get_variable('w6_1', [dim5, dim6], initializer=w_init)

        z1 = tf.matmul(input, w4)
        z1 = tf.matmul(z1, w5)
        z_mean = tf.matmul(z1, w6)
        z_variance = tf.nn.softplus(tf.matmul(z1, w6_1))

    return z_mean,z_variance

def CIFAR_SubEncoder_2(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        w_init = tf.glorot_uniform_initializer()
        dim1 = 2000
        dim2 = 1500
        dim3 = 1000
        dim4 = 600
        dim5 = 300
        dim6 = 200

        w3 = tf.get_variable('3', [dim2, dim3], initializer=w_init)
        w4 = tf.get_variable('w4', [dim3, dim4], initializer=w_init)
        w5 = tf.get_variable('w5', [dim4, dim5], initializer=w_init)
        w6 = tf.get_variable('w6', [dim5, dim6], initializer=w_init)
        w6_1 = tf.get_variable('w6_1', [dim5, dim6], initializer=w_init)

        z1 = tf.matmul(input, w3)
        z1 = tf.matmul(z1, w4)
        z1 = tf.matmul(z1, w5)
        z_mean = tf.matmul(z1, w6)
        z_variance = tf.nn.softplus(tf.matmul(z1, w6_1))

    return z_mean,z_variance


def BatchEnsemble_CIFAR_SuperGenerator(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        w_init = tf.glorot_uniform_initializer()
        dim1 = 2000
        dim2 = 1500
        dim3 = 1000
        dim4 = 600
        dim5 = 300
        dim6 = 200

        g1 = tf.get_variable('g1', [dim6, dim5], initializer=w_init)
        g2 = tf.get_variable('g2', [dim5, dim4], initializer=w_init)
        g3 = tf.get_variable('g3', [dim4, dim3], initializer=w_init)

        x = tf.matmul(input, g1)
        x = tf.matmul(x, g2)
        x = tf.matmul(x, g3)

        w_init = tf.glorot_uniform_initializer()
        w1 = tf.get_variable('w1', [2000, 32 * 32*3], initializer=w_init)

        return x,w1

def BatchEnsemble_CIFAR_SubGenerator(input, name,w1, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        w_init = tf.glorot_uniform_initializer()
        dim1 = 2000
        dim2 = 1500
        dim3 = 1000
        dim4 = 600
        dim5 = 300
        dim6 = 200
        imageDim = 32*32*3

        g4 = tf.get_variable('g4', [dim3, dim2], initializer=w_init)
        g5 = tf.get_variable('g5', [dim2, dim1], initializer=w_init)

        g6 = tf.get_variable('g6', [dim1, imageDim], initializer=w_init)
        w_init = tf.glorot_uniform_initializer()
        a1 = tf.get_variable('w1', [1, dim1], initializer=w_init)
        a2 = tf.get_variable('w2', [1, 32 * 32*3], initializer=w_init)
        w_specific = tf.matmul(tf.transpose(a1), a2)

        x = tf.matmul(input, g4)
        x = tf.matmul(x, g5)

        final_w = w1 * w_specific
        out = tf.matmul(x, final_w)
        out = tf.nn.sigmoid(out)

        #x = tf.matmul(x, g6)
        #x = tf.nn.sigmoid(x)

    return out

def CIFAR_SuperGenerator_Deep(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            z = input
            kernel = 3
            myScale = 1
            # fully-connected layers
            net = lrelu(bn(linear(input, 300, scope='c_fc4'), is_training=is_training, scope='c_bn4'))
            net = lrelu(bn(linear(net, 600, scope='c_fc2'), is_training=is_training, scope='c_bn2'))
            net = lrelu(bn(linear(net, 1500, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
            #z_dim = 1000
            z_dim = 2000
            z_mean = lrelu(bn(linear(net, z_dim, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        return z_mean

def TotalEncoder32(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_dim = 256
        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

    return z_mean, z_log_sigma_sq


def Discriminator_Celeba_Orginal(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        z_mix = tf.nn.sigmoid(z_logits)
        return z_mix,z_logits


def CIFAR_SuperGenerator(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        w_init = tf.glorot_uniform_initializer()
        dim1 = 2000
        dim2 = 1500
        dim3 = 1000
        dim4 = 600
        dim5 = 300
        dim6 = 200

        g1 = tf.get_variable('g1', [dim6, dim5], initializer=w_init)
        g2 = tf.get_variable('g2', [dim5, dim4], initializer=w_init)
        g3 = tf.get_variable('g3', [dim4, dim3], initializer=w_init)

        x = tf.matmul(input, g1)
        x = tf.matmul(x, g2)
        x = tf.matmul(x, g3)

        return x

def CIFAR_SubGenerator_Deep(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):

            #net = lrelu(bn(linear(input, 600, scope='c_fc2'), is_training=is_training, scope='c_bn2'))
            #net = lrelu(bn(linear(input, 1000, scope='c_fc6'), is_training=is_training, scope='c_bn6'))
            net = lrelu(bn(linear(input, 2000, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
            net = lrelu(bn(linear(net, 3000, scope='c_fc7'), is_training=is_training, scope='c_bn7'))
            output = linear(net, 32 * 32 * 3, scope='c_fc4')
            output = tf.nn.sigmoid(output)

        return output


def CIFAR_SubGenerator_Deep_84(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):

            #net = lrelu(bn(linear(input, 600, scope='c_fc2'), is_training=is_training, scope='c_bn2'))
            #net = lrelu(bn(linear(input, 1000, scope='c_fc6'), is_training=is_training, scope='c_bn6'))
            net = lrelu(bn(linear(input, 2000, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
            net = lrelu(bn(linear(net, 3000, scope='c_fc7'), is_training=is_training, scope='c_bn7'))
            output = linear(net, 84 * 84 * 3, scope='c_fc4')
            output = tf.nn.sigmoid(output)

        return output


def CIFAR_SubGenerator(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        w_init = tf.glorot_uniform_initializer()
        dim1 = 2000
        dim2 = 1500
        dim3 = 1000
        dim4 = 600
        dim5 = 300
        dim6 = 200
        imageDim = 32*32*3

        g4 = tf.get_variable('g4', [dim3, dim2], initializer=w_init)
        g5 = tf.get_variable('g5', [dim2, dim1], initializer=w_init)
        g6 = tf.get_variable('g6', [dim1, imageDim], initializer=w_init)

        x = tf.matmul(input, g4)
        feature = x
        x = tf.matmul(x, g5)
        x = tf.matmul(x, g6)
        x = tf.nn.sigmoid(x)

    return x,feature

def CIFAR_SubGenerator_84(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        w_init = tf.glorot_uniform_initializer()
        dim1 = 2000
        dim2 = 1500
        dim3 = 1000
        dim4 = 600
        dim5 = 300
        dim6 = 200
        imageDim = 84*84*3

        g4 = tf.get_variable('g4', [dim3, dim2], initializer=w_init)
        g5 = tf.get_variable('g5', [dim2, dim1], initializer=w_init)
        g6 = tf.get_variable('g6', [dim1, imageDim], initializer=w_init)

        x = tf.matmul(input, g4)
        feature = x
        x = tf.matmul(x, g5)
        x = tf.matmul(x, g6)
        x = tf.nn.sigmoid(x)

    return x,feature


def CIFAR_SubGenerator_Tanh(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        w_init = tf.glorot_uniform_initializer()
        dim1 = 2000
        dim2 = 1500
        dim3 = 1000
        dim4 = 600
        dim5 = 300
        dim6 = 200
        imageDim = 32*32*3

        g4 = tf.get_variable('g4', [dim3, dim2], initializer=w_init)
        g5 = tf.get_variable('g5', [dim2, dim1], initializer=w_init)
        g6 = tf.get_variable('g6', [dim1, imageDim], initializer=w_init)

        x = tf.matmul(input, g4)
        feature = x
        x = tf.matmul(x, g5)
        x = tf.matmul(x, g6)
        x = tf.nn.tanh(x)

    return x,feature


def MNIST_SuperEncoder(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        input = tf.reshape(input,(-1,28*28))

        z_dim = 200
        #is_training = True
        net = lrelu(bn(linear(input, 500, scope='c_fc3'), is_training=is_training, scope='c_bn3'))
        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

    return z_mean, z_log_sigma_sq


def MNIST_SubEncoder(input, name, is_training, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z_dim = 100
        net = input
        net = lrelu(bn(linear(net, 500, scope='c_fc4'), is_training=is_training, scope='c_bn4'))
        net = lrelu(bn(linear(net, 500, scope='c_fc1'), is_training=is_training, scope='c_bn1'))

        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

    return z_mean, z_log_sigma_sq

def MNIST_SuperGenerator(input, name, is_training, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z_dim = 600
        # is_training = True
        net = lrelu(bn(linear(input, 500, scope='c_fc3'), is_training=is_training, scope='c_bn3'))
        net = lrelu(bn(linear(net, 500, scope='c_fc1'), is_training=is_training, scope='c_bn1'))

        z_mean = linear(net, z_dim, 'e_mean')
    return z_mean


def MNIST_SubGenerator(input, name, is_training, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z_dim = 100
        net = input
        net = lrelu(bn(linear(net, 500, scope='c_fc4'), is_training=is_training, scope='c_bn4'))
        out = linear(net, 28*28, scope='g_fc3')
        out = tf.nn.sigmoid(out)
    return out

def BatchEnsemble_MNIST_SuperGenerator(input, name, is_training, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z_dim = 600
        # is_training = True
        net = lrelu(bn(linear(input, 500, scope='c_fc3'), is_training=is_training, scope='c_bn3'))
        net = lrelu(bn(linear(net, 500, scope='c_fc1'), is_training=is_training, scope='c_bn1'))

        z_mean = linear(net, z_dim, 'e_mean')

        w_init = tf.glorot_uniform_initializer()
        w1 = tf.get_variable('w1', [z_dim,28*28], initializer=w_init)

    return z_mean,w1

def BatchEnsemble_MNIST_SubGenerator(input, name,w1, is_training, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_init = tf.glorot_uniform_initializer()
        a1 = tf.get_variable('w1', [1,600], initializer=w_init)
        a2 = tf.get_variable('w2', [1,28*28], initializer=w_init)
        w_specific = tf.matmul(tf.transpose(a1),a2)

        #w_specific = tf.transpose(w_specific)

        final_w = w1 * w_specific
        out = tf.matmul(input,final_w)
        out = tf.nn.sigmoid(out)
    return out


def AuxilaryBenchmark_MNIST_Encoder_Small_a(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        input = tf.reshape(input,(-1,28*28))

        z_dim = 50
        #is_training = True
        net = lrelu(bn(linear(input, 500, scope='c_fc3'), is_training=is_training, scope='c_bn3'))
        net = lrelu(bn(linear(net, 500, scope='c_fc4'), is_training=is_training, scope='c_bn4'))

        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

    return z_mean,z_log_sigma_sq

def Encoder_SVHN_Specific_Supervised_3(image, name, batch_size=64, reuse=False):
    is_training = True
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256

        kernelSize=1
        net = slim.fully_connected(image, 1024 * kernelSize, scope='fc3')
        net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default

        h5 = linear(image, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 10
        logoutput = linear(h5, continous_len, 'e_log_sigma_sq')
        softmaxValue = tf.nn.softmax(logoutput)

        return logoutput,softmaxValue,h5


def Encoder_SVHN_Specific_Supervised_2(image, name, batch_size=64, reuse=False):
    is_training = True
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256

        kernelSize=1
        net = slim.fully_connected(image, 1024 * kernelSize, scope='fc3')
        net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default

        h5 = linear(image, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 10
        logoutput = linear(h5, continous_len, 'e_log_sigma_sq')
        softmaxValue = tf.nn.softmax(logoutput)

        return logoutput,softmaxValue


def Encoder_SVHN_Specific_Supervised_2_Cifar100(image, name, batch_size=64, reuse=False):
    is_training = True
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256

        kernelSize=1
        net = slim.fully_connected(image, 1024 * kernelSize, scope='fc3')
        net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default

        h5 = linear(image, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 10
        logoutput = linear(h5, continous_len, 'e_log_sigma_sq')
        softmaxValue = tf.nn.softmax(logoutput)

        return logoutput,softmaxValue

def Encoder_SVHN_Specific_Supervised_2_Cifar100_3(image, name, batch_size=64, reuse=False):
    is_training = True
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256

        kernelSize=1
        net = slim.fully_connected(image, 1024 * kernelSize, scope='fc3')
        net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default

        h5 = linear(image, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 10
        logoutput = linear(h5, continous_len, 'e_log_sigma_sq')
        softmaxValue = tf.nn.softmax(logoutput)

        return logoutput,softmaxValue,h5


def Encoder_SVHN_Specific_Supervised_2_Cifar100_Special(image, name, batch_size=64, reuse=False):
    is_training = True
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256

        kernelSize=1
        net = slim.fully_connected(image, 1024 * kernelSize, scope='fc3')
        net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default

        h5 = linear(image, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 20
        logoutput = linear(h5, continous_len, 'e_log_sigma_sq')
        softmaxValue = tf.nn.softmax(logoutput)

        return logoutput,softmaxValue


def Encoder_SVHN_Shared_Supervised_2_Cifar100(image,y,name, batch_size=64, reuse=False):
    is_training = True
    with tf.variable_scope(name, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            x = tf.reshape(image, [-1, 32, 32, 3])

            # For slim.conv2d, default argument values are like
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # padding='SAME', activation_fn=nn.relu,
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            kernelSize = 2
            net = slim.conv2d(x, 32*kernelSize, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64*kernelSize, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.conv2d(net, 128*kernelSize, [5, 5], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.conv2d(net, 256*kernelSize, [5, 5], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.flatten(net, scope='flatten3')

            # For slim.fully_connected, default argument values are like
            # activation_fn = nn.relu,
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            #outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return net



def AuxilaryBenchmark_CIFAR10_Encoder_Small_a(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        #input = tf.reshape(input,(64,16,49))
        #input = tf.transpose(input, [0, 2,1])
        embeddim = 100
        z_dim = embeddim
        #is_training = True
        net = lrelu(bn(linear(input, 400, scope='c_fc3'), is_training=is_training, scope='c_bn3'))
        net = lrelu(bn(linear(net, 200, scope='c_fc2'), is_training=is_training, scope='c_bn2'))
        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

    return z_mean,z_log_sigma_sq


def AuxilaryBenchmark_CIFAR10_Encoder_Small_a_Small(input, name, is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        #input = tf.reshape(input,(64,16,49))
        #input = tf.transpose(input, [0, 2,1])
        embeddim = 100
        z_dim = embeddim
        #is_training = True
        net = lrelu(bn(linear(input, 400, scope='c_fc3'), is_training=is_training, scope='c_bn3'))
        net = lrelu(bn(linear(net, 200, scope='c_fc2'), is_training=is_training, scope='c_bn2'))
        z_mean = linear(net, z_dim, 'e_mean')
        #z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        #z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

    return z_mean,z_mean


def AuxilaryBenchmark_MNIST_Encoder_Small_z(input, name,is_training , batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z_dim = 50
        #is_training = True
        net = lrelu(bn(linear(input, 500, scope='c_fc3'), is_training=is_training, scope='c_bn3'))
        net = lrelu(bn(linear(net, 200, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

    return z_mean,z_log_sigma_sq

def Benchmark_MNIST_SharedEncoder_Small_CNN(input,name,is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        input = tf.reshape(input,(-1,28*28))
        #is_training = True

        net = lrelu(bn(linear(input, 500, scope='c_fc3'), is_training=is_training, scope='c_bn3'))
        net = lrelu(bn(linear(net, 500, scope='c_fc4'), is_training=is_training, scope='c_bn4'))
    return net


def Benchmark_MNIST_SharedEncoder_Small(input,name,is_training,batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        input = tf.reshape(input,(-1,28*28))
        #is_training = True
        '''
        net = lrelu(bn(linear(input, 500, scope='c_fc3'), is_training=is_training, scope='c_bn3'))
        net = lrelu(bn(linear(net, 500, scope='c_fc4'), is_training=is_training, scope='c_bn4'))
        '''
        net = linear(input, 200, scope='c_fc3')
        net = tf.nn.tanh(net)
        net = linear(net, 200, scope='c_fc4')
        net = tf.nn.sigmoid(net)
    return net

def Benchmark_MNIST_SpecificEncoder_Small_CNN(input, name,is_training, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z_dim = 50
        #is_training = True
        net = lrelu(bn(linear(input, 500, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 200, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(input, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

    return z_mean,z_log_sigma_sq

def Benchmark_MNIST_SpecificEncoder_Small(input, name,is_training, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z_dim = 100
        '''
        #is_training = True
        net = lrelu(bn(linear(input, 500, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 200, scope='e_fc11'), is_training=is_training, scope='c_bn11'))
        '''
        z_mean = linear(input, z_dim, 'e_mean')
        z_log_sigma_sq = linear(input, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.exp(z_log_sigma_sq)

    return z_mean,z_log_sigma_sq


def Benchmark_MNIST_SharedGenerator_Small(name, input,is_training, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

            # merge noise and code
        #is_training = True
        batch_size = 64
        #net = tf.nn.relu(bn(linear(input, 500, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
        #net = tf.nn.relu(bn(linear(net, 500, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
        net = linear(input, 500, scope='c_fc3')
        net = tf.nn.tanh(net)

    return net

def Benchmark_MNIST_SpecificGenerator_Small(name, input,is_training, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        #is_training = True
        batch_size = 64
        #net = tf.nn.relu(bn(linear(input, 500, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
        net = linear(input, 500, scope='c_fc3')
        net = tf.nn.tanh(net)
        out = linear(net, 28*28, scope='g_fc3')
        #out = tf.nn.sigmoid(linear(net, 28*28, scope='g_fc3'))

    return out

def Benchmark_MNIST_SharedEncoder(input,name,batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        e = tf.reshape(input, [-1, 28, 28, 1],name="a1")
        e = conv2d(e, 32, 5,5,2,2,name="a2")
        e = conv2d(e, 64, 5,5,2,2,name="a3")
        e = conv2d(e, 128, 3,3,2,2,name="a4")
        e = tf.reshape(e,(batch_size,-1))

    return e

def Benchmark_MNIST_SpecificEncoder(input, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z_dim = 100
        is_training = True
        net = lrelu(bn(linear(input, 500, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 200, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

    return z_mean,z_log_sigma_sq

def Benchmark_MNIST_SharedGenerator_(name, input,is_training, batch_size, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()


            # merge noise and code
        #is_training = True
        net = tf.nn.relu(bn(linear(input, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
        net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))

    return net

def Benchmark_MNIST_SharedGenerator_Small_255(name, input,is_training, batch_size, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

            # merge noise and code
        #is_training = True
        #batch_size = 64
        net = tf.nn.relu(bn(linear(input, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
        net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))

    return net

def Benchmark_MNIST_SharedGenerator_Small_(name, input,is_training, batch_size, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
            # merge noise and code
        #is_training = True
        #net = tf.nn.relu(bn(linear(input, 500, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
        net = linear(input, 200, scope='c_fc3')
        net = tf.nn.tanh(net)
    return net

def Benchmark_MNIST_SharedGenerator_Small_CNN(name, input,is_training, batch_size, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
            # merge noise and code
        #is_training = True
        net = tf.nn.relu(bn(linear(input, 500, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
    return net


def Benchmark_MNIST_SharedGenerator(name, input,is_training,batch_size, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

            # merge noise and code
        #is_training = True
        #batch_size = 64
        net = tf.nn.relu(bn(linear(input, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
        net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))

    return net

def Benchmark_MNIST_SpecificGenerator_Small_(name, input,is_training,batch_size, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        #net = tf.nn.relu(bn(linear(input, 500, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
        net = linear(input, 200, scope='c_fc3')
        net = tf.nn.tanh(net)
        out = linear(net, 28*28, scope='g_fc3')
    return out

def Benchmark_MNIST_SpecificGenerator_Small_255(name, input,is_training,batch_size, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        #is_training = True
        #batch_size = 64
        net = tf.reshape(input, [batch_size, 7, 7, 128])
        net = tf.nn.relu(
            bn(deconv2d(net, [batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
               scope='g_bn3'))

        #out = tf.nn.sigmoid(deconv2d(net, [batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))
        out = deconv2d(net, [batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4')

    return out

def Benchmark_MNIST_SpecificGenerator_Small_CNN(name, input,is_training,batch_size, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

            # is_training = True
            # batch_size = 64
            # net = tf.nn.relu(bn(linear(input, 500, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
        net = linear(input, 500, scope='c_fc3')
        net = tf.nn.tanh(net)
        out = linear(net, 28 * 28, scope='g_fc3')
        # out = tf.nn.sigmoid(linear(net, 28*28, scope='g_fc3'))

    return out


def Benchmark_MNIST_SpecificGenerator_(name, input,is_training,batch_size, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        #is_training = True
        #batch_size = 64
        net = tf.reshape(input, [batch_size, 7, 7, 128])
        net = tf.nn.relu(
            bn(deconv2d(net, [batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
               scope='g_bn3'))

        #out = tf.nn.sigmoid(deconv2d(net, [batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))
        out = deconv2d(net, [batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4')

    return out


def Benchmark_MNIST_SpecificGenerator(name, input,is_training,batch_size, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        #is_training = True
        #batch_size = 64
        net = tf.reshape(input, [batch_size, 7, 7, 128])
        net = tf.nn.relu(
            bn(deconv2d(net, [batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
               scope='g_bn3'))

        #out = tf.nn.sigmoid(deconv2d(net, [batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))
        out = deconv2d(net, [batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4')

    return out

def Generator_SVHN_LGM(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        myScale = 2
        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, myScale * 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256 * myScale])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256 * myScale],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256 * myScale],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256 * myScale],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256 * myScale],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

def Generator_Celeba_Sigmoid(name, z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        '''
        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h6, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.sigmoid(h8)

        return h8

def Generator_Celeba_Tanh_84(name, z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        '''
        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h6, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8


def Generator_Celeba_Tanh(name, z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        '''
        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h6, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8


def Generator_SVHN(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.sigmoid(h8)

        return h8

def Generator_SVHN_Tiny(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 64 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 64])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 64],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 64],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 64],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 64],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.sigmoid(h8)

        return h8


def Generator_SVHN_28(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h3, [batch_size, 28, 28, 1],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.sigmoid(h8)

        return h8

def Generator_SVHN_Sigmoid(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.sigmoid(h8)

        return h8


def Generator_SVHN_Small2(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.sigmoid(h8)

        return h8


def Generator_SVHN_Small(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.sigmoid(h8)

        return h8


def Generator_SVHN_Tanh(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8



def Generator_SVHN_Sizeable(name,z_in,size, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        if size == 1:
            kernelN = 128
        elif size == -1:
            kernelN = 64
        elif size == -2:
            kernel = 32
        elif size == 2:
            kernelN = 256
        elif size == 3:
            kernelN = 512

        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, kernelN * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, kernelN])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, kernelN],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, kernelN],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, kernelN],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, kernelN],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

def Generator_SharedSVHN(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        return h3

def Generator_SharedSVHN_Small(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 128 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 128])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 128],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 128],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        return h3


def Generator_SharedSVHN2_BatchEnsemble(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        return h4

def Generator_SubSVHN2_BatchEnsemble(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 6
        h3 = z_in

        h5 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

def Generator_SharedSVHN2(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        return h5

def Generator_SubSVHN2(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        h3 = z_in

        h8 = deconv2d(h3, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

def Generator_SubSVHN(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        h3 = z_in

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.sigmoid(h8)

        return h8

def Generator_SubSVHN_Sigmoid(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        h3 = z_in

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.sigmoid(h8)

        return h8


def Generator_SubSVHN_256(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        h3 = z_in

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        #h8 = tf.nn.tanh(h8)

        return h8


def Generator_SubSVHN_Small(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        h3 = z_in

        h4 = deconv2d(h3, [batch_size, 32, 32, 128],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 128],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

def Discriminator_SVHN(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.keras.layers.batch_norm(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        z_mix = tf.nn.sigmoid(z_logits)
        return z_mix, z_logits,h5

def Discriminator_SVHN_Logits(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(d_bn5(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        return z_logits

def Discriminator_SVHN_Sigmoid(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(d_bn5(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        z_logits = tf.nn.sigmoid(z_logits)

        return z_logits


def Encoder_SVHN_Small(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))
        h2 = tf.reshape(h2, [batch_size, -1])

        h5 = linear(h2, 500, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 100
        z_mean = linear(h5, continous_len, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq


def Encoder_SVHN(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 200
        z_mean = linear(h5, continous_len, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq

def Encoder_SVHN_Tiny(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 128, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 256, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 500, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 200
        z_mean = linear(h5, continous_len, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq


def Encoder_SVHN_Sizeable(image, name,size,batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        if size == 2:
            kernelN = 64
            lateSize = 1024
        elif size == 1:
            kernelN = 32
            lateSize = 512
        elif size == -1:
            kernelN = 16
            lateSize = 256
        elif size == -2:
            kernelN = 8
            lateSize = 128
        elif size == 3:
            kernelN = 128
            lateSize = 1024

        batch_size = 64
        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, kernelN*2, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, kernelN*4, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, kernelN*6, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, lateSize, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 100
        z_mean = linear(h5, 100, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq

def Encoder_SVHN_LGM(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256

        myScale = 2
        h1 = conv2d(image, 64*myScale, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 128*myScale, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256*myScale, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512*myScale, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024*myScale, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 100
        z_mean = linear(h5, 100, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq

def Encoder_SVHN_2_Small(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 64, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 128, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 128, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 200, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 200
        z_mean = linear(h5, 200, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq,h5


def Encoder_SVHN_2(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 100
        z_mean = linear(h5, 100, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq,h5

def Encoder_SVHN_Shared(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        size = 2
        z_dim = 256
        h1 = conv2d(image, 64*size, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 128*size, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256*size, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512*size, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        return h4

def Encoder_SVHN_Shared_Small(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        size = 2
        z_dim = 256

        h1 = conv2d(image, 32*size, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 64*size, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 128*size, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 256*size, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        return h4


def Encoder_SVHN_Shared_Big(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        size = 2
        z_dim = 256
        mySize = 2
        h1 = conv2d(image, mySize*64*size, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, mySize*128*size, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, mySize*256*size, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, mySize*512*size, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        return h4

def Encoder_SVHN_Shared_BE(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        size = 2
        z_dim = 256
        kernelCount = 2
        h1 = conv2d(image, 64*kernelCount*size, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 128*kernelCount*size, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256*kernelCount*size, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512*kernelCount*size, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        return h4

def Encoder_SVHN_Specific(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        kernelCount = 1
        h5 = linear(image, 1024*kernelCount, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 100
        z_mean = linear(h5, 100, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean,z_log_sigma_sq,h5

def Encoder_SVHN_Specific_Small(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        kernelCount = 1
        h5 = linear(image, 512*kernelCount, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 100
        z_mean = linear(h5, 100, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean,z_log_sigma_sq,h5


def Encoder_SVHN_Specific_Big(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        kernelCount = 2
        h5 = linear(image, 1024*kernelCount, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 100
        z_mean = linear(h5, 100, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean,z_log_sigma_sq,h5

def Encoder_SVHN_Specific_BE(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 6
        z_dim = 256
        kernelCount = 3
        h5 = linear(image, 1024*kernelCount, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 100
        z_mean = linear(h5, 100, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean,z_log_sigma_sq,h5

def Encoder_SVHN_Shared_Supervised(image,y,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])
        h4 = tf.concat((h4,y),axis=1)

        return h4

def Encoder_SVHN_Supervised_(image,y,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])
        h4 = tf.concat((h4,y),axis=1)

        batch_size = 64
        kernel = 3
        z_dim = 256
        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 10
        logoutput = linear(h5, continous_len, 'e_log_sigma_sq')
        softmaxValue = tf.nn.softmax(logoutput)

        return logoutput, softmaxValue

def Discriminator_SVHN_WGAN(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        return z_logits

def Encoder_SVHN_Shared_Supervised_2(image,y,name, batch_size=64, reuse=False):
    is_training = True
    with tf.variable_scope(name, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            x = tf.reshape(image, [-1, 32, 32, 3])

            # For slim.conv2d, default argument values are like
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # padding='SAME', activation_fn=nn.relu,
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            kernelSize = 2
            net = slim.conv2d(x, 32*kernelSize, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64*kernelSize, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')

            # For slim.fully_connected, default argument values are like
            # activation_fn = nn.relu,
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            #outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return net


def Discriminator_SVHN_WGAN_28(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        return z_logits


def Encoder_SVHN_Shared_Supervised2(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        bc = 1
        h1 = conv2d(image, bc*32, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, bc*64, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, bc*128, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, bc*256, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        return h4

def Encoder_SVHN_Specific_Supervised(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        h5 = linear(image, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 10
        logoutput = linear(h5, continous_len, 'e_log_sigma_sq')
        softmaxValue = tf.nn.softmax(logoutput)

        return logoutput,softmaxValue

def Encoder_SVHN2(inputs,scopename, is_training=True,reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            #x = tf.reshape(inputs, [-1, 28, 28, 1])
            x = tf.reshape(inputs, [-1, 32, 32, 3])

            # For slim.conv2d, default argument values are like
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # padding='SAME', activation_fn=nn.relu,
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.conv2d(x, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')

            # For slim.fully_connected, default argument values are like
            # activation_fn = nn.relu,
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.fully_connected(net, 1024, scope='fc3')
            net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
            outputs = slim.fully_connected(net, 4, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs,tf.nn.softmax(outputs)

def Generator_mnist(name,z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # ArchitecDiscriminator_Celebature : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope(name, reuse=reuse):

            # merge noise and code
            batch_size = 64
            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [batch_size, 7, 7, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
                   scope='g_bn3'))

            out = tf.nn.sigmoid(deconv2d(net, [batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

            return out

def Encoder_mnist(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 10

        is_training = True
        z_dim = 100
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))
        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq,out_logit,softmaxValue

def classifier_mnist( x, is_training=True, reuse=False):
    batch_size = 128
    len_discrete_code = 10
    with tf.variable_scope("classifier", reuse=reuse):
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
        out_logit = linear(net, len_discrete_code, scope='c_fc2')
        out = tf.nn.softmax(out_logit)

    return out, out_logit

def Discriminator_Mnist( x,name, batch_size = 64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        is_training = True
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
        out_logit = linear(net, 1, scope='d_fc4')
        out = tf.nn.sigmoid(out_logit)

    return out, out_logit, net

def Discriminator_Mnist2( x,z,name, batch_size = 64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        is_training = True
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = tf.concat((net,z),axis=1)
        net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))

        len_discrete_code = 4
        out_logit2 = linear(net, len_discrete_code, scope='d_fc22')
        softmaxValue = tf.nn.softmax(out_logit2)

        out_logit = linear(net, 1, scope='d_fc4')
        out = tf.nn.sigmoid(out_logit)

    return out, out_logit, net,out_logit2,softmaxValue

def Generator_Celeba(z,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        '''
        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h6, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8


def Generator_Celeba_Shared_BatchEnsemble(name,z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        myT = 3
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        kernel = 3
        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        kernel = 3
        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        return h6

def Generator_Celeba_Specific_BatchEnsemble(name,z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = 3
        h8 = deconv2d(z, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

def Generator_Celeba_Shared_ImageTranslation(name,image,z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        #encoding
        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        is_training = True
        len_discrete_code = 4
        net = lrelu(bn(linear(h5, 64, scope='e_fc11'), is_training=is_training, scope='c_bn1'))
        out_logit = linear(net, len_discrete_code, scope='e_fc22')


        z = tf.concat((out_logit,z),axis=1)
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        return h4

def Generator_Celeba_Specific_128(name,z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        h5 = deconv2d(z, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))


        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))

        h8 = deconv2d(h7, [batch_size, 128, 128, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

def Generator_Celeba_Shared(name,z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        return h4

def Generator_Celeba_Shared_Small(name,z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 128 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 128])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 128],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 128],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 128],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        return h4

def Generator_Celeba_Specific_Small(name,z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        h5 = deconv2d(z, [batch_size, 32, 32, 128],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        '''
        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h6, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

def Generator_Celeba_Specific(name,z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        h5 = deconv2d(z, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        '''
        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h6, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

def Encoder_Celeba(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        is_training = True
        len_discrete_code = 4
        net = lrelu(bn(linear(h5, 64, scope='e_fc11'), is_training=is_training, scope='c_bn1'))
        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq,out_logit,softmaxValue

def Encoder_Celeba_84(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        is_training = True
        len_discrete_code = 4
        net = lrelu(bn(linear(h5, 64, scope='e_fc11'), is_training=is_training, scope='c_bn1'))

        feature = net

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq,feature


def Encoder_Celeba_Shared_LGM(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        size = 3
        z_dim = 256
        h1 = conv2d(image, 64*size, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128*size, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256*size, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        return h3

def Encoder_Celeba_Specific_LGM(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        size = 3
        h4 = conv2d(image, 512*size, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024*size, 'e_h5_lin')
        h5 = lrelu(h5)

        is_training = True

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq,h5

def Encoder_Celeba_Shared_CURL(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        size = 2
        z_dim = 256
        h1 = conv2d(image, 64*size, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128*size, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256*size, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        return h3

def Encoder_Celeba_Specific_CURL(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        size = 2
        h4 = conv2d(image, 512*size, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024*size, 'e_h5_lin')
        h5 = lrelu(h5)

        is_training = True

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq,h5

def Encoder_Celeba_Shared(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        return h3

def Encoder_Celeba_Shared_Small(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 32, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 64, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 64, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        return h3

def Encoder_Celeba_Specific_Small(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h4 = conv2d(image, 256, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 512, 'e_h5_lin')
        h5 = lrelu(h5)

        is_training = True

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq,h5



def Encoder_Celeba_Specific(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h4 = conv2d(image, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        is_training = True

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq,h5

def Encoder_Celeba2(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        is_training = True
        len_discrete_code = 3
        net = lrelu(bn(linear(h5, 64, scope='e_fc11'), is_training=is_training, scope='c_bn1'))
        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq

def Encoder_Celeba2_classifier(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        is_training = True
        len_discrete_code = 4
        net = lrelu(bn(linear(h5, 64, scope='e_fc11'), is_training=is_training, scope='c_bn1'))
        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        return out_logit, softmaxValue

def Discriminator_Celeba(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        z_mix = tf.nn.sigmoid(z_logits)
        return z_mix,z_logits

def Discriminator_Celeba2(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        z_mix = z_logits#tf.nn.sigmoid(z_logits)
        return z_mix,z_logits
