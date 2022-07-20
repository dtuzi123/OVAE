import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras as K

import os
import gzip
import cv2
#import keras as keras
import os
import scipy.io as scio
#from Utils2 import *
#from scipy.misc import imsave as ims
from tensorflow.keras import datasets
import tensorflow.keras as K

def GiveMNIST32_Tanh():
    mnistName = "MNIST"
    data_X, data_y = load_mnist_tanh(mnistName)

    # data_X = np.expand_dims(data_X, axis=3)
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i], size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    return mnist_train_x, mnist_train_label, mnist_test, mnist_label_test


def GiveMNIST32():
    mnistName = "MNIST"
    data_X, data_y = load_mnist(mnistName)

    # data_X = np.expand_dims(data_X, axis=3)
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i], size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    return mnist_train_x, mnist_train_label, mnist_test, mnist_label_test


def Give_InverseFashion32_Tanh():
    mnistName = "Fashion"
    data_X, data_y = load_mnist_tanh(mnistName)
    data_X = np.reshape(data_X, (-1, 28, 28))
    for i in range(np.shape(data_X)[0]):
        for k1 in range(28):
            for k2 in range(28):
                data_X[i, k1, k2] = 1.0 - data_X[i, k1, k2]

    data_X = np.reshape(data_X, (-1, 28, 28, 1))
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)
    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i], size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test


def Give_InverseFashion32():
    mnistName = "Fashion"
    data_X, data_y = load_mnist(mnistName)
    data_X = np.reshape(data_X, (-1, 28, 28))
    for i in range(np.shape(data_X)[0]):
        for k1 in range(28):
            for k2 in range(28):
                data_X[i, k1, k2] = 1.0 - data_X[i, k1, k2]

    data_X = np.reshape(data_X, (-1, 28, 28, 1))
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)
    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i], size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test


def ReturnSet_ByIndex(x,y,startIndex,endIndex):
    xarr = []
    yarr = []
    difference = endIndex - 10
    for i in range(np.shape(x)[0]):
        if y[i] >= startIndex and y[i] <= endIndex:
            xarr.append(x[i])
            label = y[i] - difference
            label = label-1
            yarr.append(label)

    xarr = np.array(xarr)
    yarr = np.array(yarr)
    return xarr,yarr

def ReturnSet_ByIndex2(x,y,startIndex,endIndex):
    xarr = []
    yarr = []
    difference = endIndex - 20
    for i in range(np.shape(x)[0]):
        if y[i] >= startIndex and y[i] <= endIndex:
            xarr.append(x[i])
            label = y[i] - difference
            label = label-1
            yarr.append(label)

    xarr = np.array(xarr)
    yarr = np.array(yarr)
    return xarr,yarr

def ReturnSet_ByIndex(x,y,startIndex,endIndex):
    xarr = []
    yarr = []
    difference = endIndex - 10
    for i in range(np.shape(x)[0]):
        if y[i] >= startIndex and y[i] <= endIndex:
            xarr.append(x[i])
            label = y[i] - difference
            label = label-1
            yarr.append(label)

    xarr = np.array(xarr)
    yarr = np.array(yarr)
    return xarr,yarr

def Split_CIFAR100_ReturnTesting():

    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

    x_train = x_train/255
    x_test = x_test/ 255


    x1_,y1_ = ReturnSet_ByIndex(x_test,y_test,1,10)
    x2_,y2_ = ReturnSet_ByIndex(x_test,y_test,11,20)
    x3_,y3_ = ReturnSet_ByIndex(x_test,y_test,21,30)
    x4_,y4_ = ReturnSet_ByIndex(x_test,y_test,31,40)
    x5_,y5_ = ReturnSet_ByIndex(x_test,y_test,41,50)

    '''
    y1_ = to_categorical(y1_, num_classes=None)
    y2_ = to_categorical(y2_, num_classes=None)
    y3_ = to_categorical(y3_, num_classes=None)
    y4_ = to_categorical(y4_, num_classes=None)
    y5_ = to_categorical(y5_, num_classes=None)
    '''

    return x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_,x5_,y5_

def Split_CIFAR100_ReturnTesting_Special():
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

    x_train = x_train/255
    x_test = x_test/ 255

    #from keras.utils.np_utils import to_categorical

    x1_,y1_ = ReturnSet_ByIndex(x_test,y_test,1,20)
    x2_,y2_ = ReturnSet_ByIndex(x_test,y_test,21,40)
    x3_,y3_ = ReturnSet_ByIndex(x_test,y_test,41,60)
    x4_,y4_ = ReturnSet_ByIndex(x_test,y_test,61,80)
    x5_,y5_ = ReturnSet_ByIndex(x_test,y_test,81,100)

    y1_ = K.utils.to_categorical(y1_, num_classes=None)
    y2_ = K.utils.to_categorical(y2_, num_classes=None)
    y3_ = K.utils.to_categorical(y3_, num_classes=None)
    y4_ = K.utils.to_categorical(y4_, num_classes=None)
    y5_ = K.utils.to_categorical(y5_, num_classes=None)

    return x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_,x5_,y5_


def Split_CIFAR100():
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

    x_train = x_train/255
    x_test = x_test/ 255
    x1,y1 = ReturnSet_ByIndex(x_train,y_train,1,10)
    x2,y2 = ReturnSet_ByIndex(x_train,y_train,11,20)
    x3,y3 = ReturnSet_ByIndex(x_train,y_train,21,30)
    x4,y4 = ReturnSet_ByIndex(x_train,y_train,31,40)
    x5,y5 = ReturnSet_ByIndex(x_train,y_train,41,50)

    x1_,y1_ = ReturnSet_ByIndex(x_test,y_test,1,10)
    x2_,y2_ = ReturnSet_ByIndex(x_test,y_test,11,20)
    x3_,y3_ = ReturnSet_ByIndex(x_test,y_test,21,30)
    x4_,y4_ = ReturnSet_ByIndex(x_test,y_test,31,40)
    x5_,y5_ = ReturnSet_ByIndex(x_test,y_test,41,50)

    x_ = np.concatenate((x1_,x2_,x3_,x4_,x5_),axis=0)
    y_ = np.concatenate((y1_,y2_,y3_,y4_,y5_),axis=0)

    return x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x_,y_

def Split_CIFAR100_2():
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

    x_train = x_train/255
    x_test = x_test/ 255
    x1,y1 = ReturnSet_ByIndex2(x_train,y_train,1,20)
    x2,y2 = ReturnSet_ByIndex2(x_train,y_train,21,40)
    x3,y3 = ReturnSet_ByIndex2(x_train,y_train,41,60)
    x4,y4 = ReturnSet_ByIndex2(x_train,y_train,61,80)
    x5,y5 = ReturnSet_ByIndex2(x_train,y_train,81,100)

    x1_,y1_ = ReturnSet_ByIndex2(x_test,y_test,1,20)
    x2_,y2_ = ReturnSet_ByIndex2(x_test,y_test,21,40)
    x3_,y3_ = ReturnSet_ByIndex2(x_test,y_test,41,60)
    x4_,y4_ = ReturnSet_ByIndex2(x_test,y_test,61,80)
    x5_,y5_ = ReturnSet_ByIndex2(x_test,y_test,81,100)

    x_ = np.concatenate((x1_,x2_,x3_,x4_,x5_),axis=0)
    y_ = np.concatenate((y1_,y2_,y3_,y4_,y5_),axis=0)

    return x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x_,y_


def Split_CIFAR100_3():
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

    x_train = x_train/255
    x_test = x_test/ 255
    x1,y1 = ReturnSet_ByIndex2(x_train,y_train,1,20)
    x2,y2 = ReturnSet_ByIndex2(x_train,y_train,21,40)
    x3,y3 = ReturnSet_ByIndex2(x_train,y_train,41,60)
    x4,y4 = ReturnSet_ByIndex2(x_train,y_train,61,80)
    x5,y5 = ReturnSet_ByIndex2(x_train,y_train,81,100)

    x1_,y1_ = ReturnSet_ByIndex2(x_test,y_test,1,20)
    x2_,y2_ = ReturnSet_ByIndex2(x_test,y_test,21,40)
    x3_,y3_ = ReturnSet_ByIndex2(x_test,y_test,41,60)
    x4_,y4_ = ReturnSet_ByIndex2(x_test,y_test,61,80)
    x5_,y5_ = ReturnSet_ByIndex2(x_test,y_test,81,100)

    x_ = np.concatenate((x1_,x2_,x3_,x4_,x5_),axis=0)
    y_ = np.concatenate((y1_,y2_,y3_,y4_,y5_),axis=0)

    return x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_,x5_,y5_


def Give_InverseDataset(name):
    data_X, data_y = load_mnist(name)
    data_X = np.reshape(data_X, (-1, 28, 28))
    for i in range(np.shape(data_X)[0]):
        for k1 in range(28):
            for k2 in range(28):
                data_X[i, k1, k2] = 1.0 - data_X[i, k1, k2]

    data_X = np.reshape(data_X,(-1,28*28))
    return data_X,data_y

def GiveLifelongTasks_AcrossDomain():
    (train_images_nonbinary, y_train), (test_images_nonbinary, y_test) = tf.keras.datasets.mnist.load_data()

    train_images_nonbinary = train_images_nonbinary.reshape(train_images_nonbinary.shape[0], 28 * 28)
    test_images_nonbinary = test_images_nonbinary.reshape(test_images_nonbinary.shape[0], 28 * 28)

    '''
    y_train = tf.cast(y_train, tf.int64)
    y_test = tf.cast(y_test, tf.int64)
    '''

    train_images = train_images_nonbinary / 255.
    test_images = test_images_nonbinary / 255.

    '''
    # Binarization
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.
    '''

    mnistTrain = train_images
    mnistTest = test_images

    mnistName = "Fashion"
    data_X, data_y = load_mnist(mnistName)

    data_X = np.reshape(data_X,(-1,28*28))

    # data_X = np.expand_dims(data_X, axis=3)
    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]

    '''
    x_train[x_train >= .5] = 1.
    x_train[x_train < .5] = 0.
    x_test[x_test >= .5] = 1.
    x_test[x_test < .5] = 0.
    '''
    fashionTrain = x_train
    fashionTest = x_test

    imnistX = Give_InverseDataset("mnist")
    ifashionX = Give_InverseDataset("Fashion")

    '''
    imnistX[imnistX >= .5] = 1.
    imnistX[imnistX < .5] = 0.
    ifashionX[ifashionX >= .5] = 1.
    ifashionX[ifashionX < .5] = 0.
    '''

    imnistTrainX = imnistX[0:60000]
    imnistTestX = imnistX[60000:70000]
    ifashionTrainX = ifashionX[0:60000]
    ifashionTestX = ifashionX[60000:70000]

    return mnistTrain,mnistTest,fashionTrain,fashionTest,imnistTrainX,imnistTestX,ifashionTrainX,ifashionTestX

def Load_Caltech101(isBinarized):
    dataFile = 'data/caltech101_silhouettes_28_split1.mat'
    data = scio.loadmat(dataFile)
    bc = 0

    trainingSet = data["train_data"]
    testingSet = data["test_data"]

    return trainingSet,testingSet

def Load_OMNIST(isBinarized):
    dataFile = 'data/omniglot.mat'
    dataFile = 'data/chardata.mat'
    data = scio.loadmat(dataFile)

    myData = data["data"]
    myData = myData.transpose(1, 0)

    #if isBinarized == True:
    #    myData[myData >= .5] = 1.
    #    myData[myData < .5] = 0.

    trainingSet = myData
    testingSet = data["testdata"]
    testingSet = testingSet.transpose(1, 0)

    return trainingSet,testingSet

'''
from utils import *
dataFile = 'data/chardata.mat'
data = scio.loadmat(dataFile)
myData = data["testdata"]
myData = myData.transpose(1, 0)
batch = myData[0:64]
batch = np.reshape(batch,(-1,28,28,1))
print(batch[0])
batch = batch * 255.0
cv2.imwrite(os.path.join("results/", 'a1.png'), merge2(batch[:64], [8, 8]))
bb = 0
'''

def load_mnist_tanh(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    X = X / 127.5 -1

    return X, y_vec


def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec


def Split_dataset_by10(x,y):
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    arr6 = []
    arr7 = []
    arr8 = []
    arr9 = []
    arr10 = []

    labelArr1 = []
    labelArr2 = []
    labelArr3 = []
    labelArr4 = []
    labelArr5 = []
    labelArr6 = []
    labelArr7 = []
    labelArr8 = []
    labelArr9 = []
    labelArr10 = []

    n = np.shape(x)[0]
    for i in range(n):
        data1 = x[i]
        label1 = y[i]
        if label1[0] == 1:
            arr1.append(data1)
            labelArr1.append(label1)

        elif label1[1] == 1:
            arr2.append(data1)
            labelArr2.append(label1)

        elif label1[2] == 1:
            arr3.append(data1)
            labelArr3.append(label1)
        elif label1[3] == 1:
            arr4.append(data1)
            labelArr4.append(label1)
        elif label1[4] == 1:
            arr5.append(data1)
            labelArr5.append(label1)
        elif label1[5] == 1:
            arr6.append(data1)
            labelArr6.append(label1)
        elif label1[6] == 1:
            arr7.append(data1)
            labelArr7.append(label1)
        elif label1[7] == 1:
            arr8.append(data1)
            labelArr8.append(label1)
        elif label1[8] == 1:
            arr9.append(data1)
            labelArr9.append(label1)
        elif label1[9] == 1:
            arr10.append(data1)
            labelArr10.append(label1)

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    arr3 = np.array(arr3)
    arr4 = np.array(arr4)
    arr5 = np.array(arr5)
    arr6 = np.array(arr6)
    arr7 = np.array(arr7)
    arr8 = np.array(arr8)
    arr9 = np.array(arr9)
    arr10 = np.array(arr10)

    labelArr1 = np.array(labelArr1)
    labelArr2 = np.array(labelArr2)
    labelArr3 = np.array(labelArr3)
    labelArr4 = np.array(labelArr4)
    labelArr5 = np.array(labelArr5)
    labelArr6 = np.array(labelArr6)
    labelArr7 = np.array(labelArr7)
    labelArr8 = np.array(labelArr8)
    labelArr9 = np.array(labelArr9)
    labelArr10 = np.array(labelArr10)


    return arr1, labelArr1, arr2, labelArr2, arr3, labelArr3, arr4, labelArr4, arr5, labelArr5,arr6, labelArr6,arr7, labelArr7,arr8, labelArr8,arr9, labelArr9,arr10, labelArr10


def Split_dataset_by5(x,y):
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    labelArr1 = []
    labelArr2 = []
    labelArr3 = []
    labelArr4 = []
    labelArr5 = []

    n = np.shape(x)[0]
    for i in range(n):
        data1 = x[i]
        label1 = y[i]
        if label1[0] == 1 or label1[1] == 1:
            arr1.append(data1)
            labelArr1.append(label1)

        if label1[2] == 1 or label1[3] == 1:
            arr2.append(data1)
            labelArr2.append(label1)

        if label1[4] == 1 or label1[5] == 1:
            arr3.append(data1)
            labelArr3.append(label1)

        if label1[6] == 1 or label1[7] == 1:
            arr4.append(data1)
            labelArr4.append(label1)

        if label1[8] == 1 or label1[9] == 1:
            arr5.append(data1)
            labelArr5.append(label1)

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    arr3 = np.array(arr3)
    arr4 = np.array(arr4)
    arr5 = np.array(arr5)

    labelArr1 = np.array(labelArr1)
    labelArr2 = np.array(labelArr2)
    labelArr3 = np.array(labelArr3)
    labelArr4 = np.array(labelArr4)
    labelArr5 = np.array(labelArr5)
    return arr1,labelArr1,arr2,labelArr2,arr3,labelArr3,arr4,labelArr4,arr5,labelArr5

def Split_dataset_by5_Specal(x,y):
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    labelArr1 = []
    labelArr2 = []
    labelArr3 = []
    labelArr4 = []
    labelArr5 = []

    n = np.shape(x)[0]
    for i in range(n):
        data1 = x[i]
        label1 = y[i]
        if label1[0] == 1 or label1[1] == 1:
            arr1.append(data1)
            labelArr1.append(label1)

        if label1[2] == 1 or label1[3] == 1:
            arr2.append(data1)
            labelArr2.append(label1)

        if label1[4] == 1 or label1[5] == 1:
            arr3.append(data1)
            labelArr3.append(label1)

        if label1[6] == 1 or label1[7] == 1:
            arr4.append(data1)
            labelArr4.append(label1)

        if label1[8] == 1 or label1[9] == 1:
            arr5.append(data1)
            labelArr5.append(label1)

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    arr3 = np.array(arr3)
    arr4 = np.array(arr4)
    arr5 = np.array(arr5)

    labelArr1 = np.array(labelArr1)
    labelArr2 = np.array(labelArr2)
    labelArr3 = np.array(labelArr3)
    labelArr4 = np.array(labelArr4)
    labelArr5 = np.array(labelArr5)
    return arr1,labelArr1,arr2,labelArr2,arr3,labelArr3,arr4,labelArr4,arr5,labelArr5
