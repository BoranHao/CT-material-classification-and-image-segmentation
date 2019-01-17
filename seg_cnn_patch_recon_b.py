#!usr/bin/env python
# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------------
'''Segmentation output for background/target. Using the results from seg_cnn_patch_l2'''

__author__      = "Boran Hao"
__date__        = "Jan, 2019"
__email__       = "brhao@bu.edu"
#-------------------------------------------------------------------------------


import warnings
# warnings.filterwarnings("ignore")
import tensorflow as tf
import pyfits
import numpy as np
import scipy.io
import random
from sklearn.metrics import confusion_matrix
import copy




def GetName(No):

    if No < 10:
        name = '00' + str(No)
    else:
        if No >= 100:
            name = str(No)
        else:
            name = '0' + str(No)
    return name


# --------------------------------------------
# Define I/O

size = 5
DataSize = 100000
path = 'D:/seg/Pdata_ds_s/'
# number=1
Ze = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1], 4: [1, 0, 0, 0]}

# tf.reset_default_graph()

batch = 40

x = tf.placeholder("float", shape=[None, size, size, size])

y_ = tf.placeholder("float", shape=[None, 2])

keep_prob = tf.placeholder(tf.float32)

# ------------------------------------------------------
# Network Construct

x_in = tf.reshape(x, [-1, size, size, size, 1])

filter = tf.Variable(tf.truncated_normal([3, 3, 3, 1, 9], stddev=0.1))
bias = tf.Variable(tf.zeros([9]))

filter2 = tf.Variable(tf.truncated_normal([3, 3, 3, 9, 24], stddev=0.1))
bias2 = tf.Variable(tf.zeros([24]))

filter3 = tf.Variable(tf.truncated_normal([3, 3, 3, 24, 32], stddev=0.1))
bias3 = tf.Variable(tf.zeros([32]))

filter4 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 96], stddev=0.1))
bias4 = tf.Variable(tf.zeros([96]))

c1 = tf.nn.relu(tf.nn.conv3d(x_in, filter, strides=[1, 1, 1, 1, 1], padding="SAME") + bias)
c2 = tf.nn.relu(tf.nn.conv3d(c1, filter2, strides=[1, 1, 1, 1, 1], padding="SAME") + bias2)
c3 = tf.nn.relu(tf.nn.conv3d(c2, filter3, strides=[1, 1, 1, 1, 1], padding="VALID") + bias3)
c4 = tf.nn.relu(tf.nn.conv3d(c3, filter4, strides=[1, 1, 1, 1, 1], padding="VALID") + bias4)

Y1 = tf.reshape(c4, [-1, 1 * 1 * 1 * 96])

W1 = tf.Variable(tf.truncated_normal([1 * 1 * 1 * 96, 64], stddev=0.1))
B1 = tf.Variable(tf.zeros([64]))

W2 = tf.Variable(tf.truncated_normal([64, 40], stddev=0.1))
B2 = tf.Variable(tf.zeros([40]))

W3 = tf.Variable(tf.truncated_normal([40, 16], stddev=0.1))
B3 = tf.Variable(tf.zeros([16]))

W4 = tf.Variable(tf.truncated_normal([16, 2], stddev=0.1))
B4 = tf.Variable(tf.zeros([2]))

'''W5 = tf.Variable(tf.truncated_normal([16, 4] ,stddev=0.1))
B5 = tf.Variable(tf.zeros([4]))'''

Y2 = tf.nn.selu(tf.matmul(Y1, W1) + B1)

Y2 = tf.nn.dropout(Y2, keep_prob)
Y3 = tf.nn.selu(tf.matmul(Y2, W2) + B2)
Y4 = tf.nn.selu(tf.matmul(Y3, W3) + B3)
#Y5 = tf.nn.selu(tf.matmul(Y4, W4) + B4)

y = tf.nn.softmax(tf.matmul(Y4, W4) + B4)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#tf.summary.scalar('loss', cross_entropy)
#cross_entropy = tf.reduce_mean(abs(y_-y))
train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#pred=tf.argmax(y, 1)
pred = tf.argmax(tf.nn.relu(y - 0.999), 1)
# pred=y

sess = tf.InteractiveSession()
#merged = tf.summary.merge_all()
tf.global_variables_initializer().run()

saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1000)

# --------------------------------------------------
# Imagefile opened

No = GetName(131)

Ifile = 'I' + str(No) + '.fits.gz'
Lfile = 'L' + str(No) + '.fits'
Sfile = 's' + str(No) + '.fits'


# LB={0:[1,0,0,0,0],1:[0,1,0,0,0],2:[0,0,1,0,0],3:[0,0,0,1,0],4:[0,0,0,0,1]}
LB = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1], 4: [1, 0, 0, 0]}
LB1 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0}

FileCT = pyfits.open('D:/seg/TestImages/' + Ifile, ignore_missing_end=True)
FileCT = FileCT[0].data
print(Ifile + ' LOADED')

FileLB = pyfits.open('D:/seg/gtl/' + Lfile, ignore_missing_end=True)
FileLB = 0 * FileLB[0].data
# FileLB=np.zeros([FileCT.shape[0],512,512],dtype=np.int16)
print(Lfile + ' LOADED')

Layer = FileLB.shape[0]
print(str(Layer) + ' Layers')


# ------------------------------
# Load Model

LoadName = 'D:/seg/PatchModels_b/Patch_CNN_Number_99'
try:
    saver.restore(sess, LoadName)
    print('Model ' + LoadName + ' successfully loaded')
except BaseException:
    print('Model ' + LoadName + ' Does not exist, parameters initialized')




def Reconstruct2():
    '''Predict the whole layer when memory is enough'''
    for i in range(200, 202):
        print(i)
        ls = [(j, k) for j in range(2, 509) for k in range(2, 509)]
        # print(ls)

        tdic = {x: np.array([FileCT[int(i - (size - 1) / 2):int(i + (size + 1) / 2),
                                    int(tu[0] - (size - 1) / 2):int(tu[0] + (size + 1) / 2),
                                    int(tu[1] - (size - 1) / 2):int(tu[1] + (size + 1) / 2)] for tu in ls]) / 65535}

        print(tdic[x].shape)
        try:
            Predict = sess.run(pred, feed_dict=tdic)
            print(Predict.shape)
            Predict = Predict.reshape(
                (int(Predict.shape[0]**0.5), int(Predict.shape[0]**0.5)))
            print(Predict.shape)
            print(Predict)
            print(FileLB)
            print(FileLB[i, 2:509, 2:509].shape)
            FileLB[i, 2:509, 2:509] = Predict
        except ValueError:
            print((i, j, k))
            continue

    Out = pyfits.open(Ifile, ignore_missing_end=True)
    Out[0].data = FileLB
    Out.writeto(Sfile)


def Reconstruct3():
    '''Separate each layers into 4 pieces when momery is limited'''
    for i in range(int((size - 1) / 2), int(FileLB.shape[0] - (size - 1) / 2)):
        # for i in [2*k for k in range(100,101)]:
        print(i)
        for xc in [(2, 256), (256, 510)]:
            for yc in [(2, 256), (256, 510)]:
                # lbond=50
                # ubond=450
                print(i)
                ls = [(j, k) for j in range(xc[0], xc[1]) for k in range(yc[0], yc[1])]
                # print(ls)

                tdic={x: np.array([FileCT[int(i-(size-1)/2):int(i+(size+1)/2), int(tu[0]-(size-1)/2):int(tu[0]+(size+1)/2), int(tu[1]-(size-1)/2):int(tu[1]+(size+1)/2)] for tu in ls])/65535, keep_prob:1}

                print(tdic[x].shape)
                # try:
                Predict = sess.run(pred, feed_dict=tdic)
                print(Predict.shape)
                Predict = Predict.reshape((int(Predict.shape[0]**0.5), int(Predict.shape[0]**0.5)))
                print(Predict.shape)
                # print(Predict)
                print(FileLB)
                # print(FileLB[i,xc[0]:xc[1],yc[0]:yc[1]].shape)
                FileLB[i, xc[0]:xc[1], yc[0]:yc[1]] = Predict
                '''except ValueError:
                    print((i,j,k))
                    continue'''

    Out = pyfits.open('D:/seg/TestImages/' + Ifile, ignore_missing_end=True)
    Out[0].data = FileLB
    print(Out[0].data)
    Out.writeto(Sfile)


def Reconstruct4():
    '''Add the likelihood of adjacent voxels to improve the segmentation smoothness'''
    Vote = np.zeros([FileLB.shape[0], 512, 512, 2])
    Tensor = 0 * copy.deepcopy(FileLB)
    # for i in range(int((size-1)/2),int(FileLB.shape[0]-(size-1)/2)):
    for i in [2 * k for k in range(100, 101)]:
        print(i)
        for xc in [(2, 256), (256, 510)]:
            for yc in [(2, 256), (256, 510)]:
                print(i)
                ls = [(j, k) for j in range(xc[0], xc[1]) for k in range(yc[0], yc[1])]

                tdic={x: np.array([FileCT[int(i-(size-1)/2):int(i+(size+1)/2), int(tu[0]-(size-1)/2):int(tu[0]+(size+1)/2), int(tu[1]-(size-1)/2):int(tu[1]+(size+1)/2)] for tu in ls])/65535, keep_prob:1}

                print(tdic[x].shape)
                Predict = sess.run(pred, feed_dict=tdic)
                print(Predict.shape)
                Predict = list(Predict)
                for n in range(len(Predict)):
                    tu = ls[n]
                    '''if np.argmax(Predict[n])==0:
                        Vote[int(i-(size-3)/2):int(i+(size-1)/2), int(tu[0]-(size-3)/2):int(tu[0]+(size-1)/2), int(tu[1]-(size-3)/2):int(tu[1]+(size-1)/2)]+=2*Predict[n]
                    else:'''
                    Vote[int(i - (size - 3) / 2):int(i + (size - 1) / 2), int(tu[0] - (size - 3) / 2):int(tu[0] + (size - 1) / 2), int(tu[1] - (size - 3) / 2):int(tu[1] + (size - 1) / 2)] += Predict[n]

    for i in [2 * k for k in range(100, 101)]:
        # piece=np.argmax(Vote[i],axis=2)
        Tensor[i] = np.argmax(Vote[i], axis=2)

    Out = pyfits.open('D:/seg/TestImages/' + Ifile, ignore_missing_end=True)
    Out[0].data = Tensor
    Out.writeto(Sfile)


with tf.device('/cpu:0'):
    Reconstruct3()
