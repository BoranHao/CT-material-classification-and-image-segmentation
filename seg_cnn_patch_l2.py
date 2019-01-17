#!usr/bin/env python
# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------------
'''Train CNN to segment background and targets using 100,000,000 samples'''

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




def GenPatchDic(size=5, Num=2, layer=-1, turn=-1000):
    array = []
    yarray = []

    for coin in range(0, 2):

        for ind in range(turn * 50, turn * 50 + 50):
            array.append(DataDic[coin][ind, :].reshape((size, size, size)))
            yarray.append(Ze[coin])

    patch = np.array(array) / 65535
    label = np.array(yarray)

    # print(label)
    #z=np.array([FileLB[tu[0],tu[1],tu[2]] for tu in array])
    dic = {x: patch, y_: label, keep_prob: 0.9}
    return dic


def GenPatchDic_test2(size=5, Num=2, layer=-1):
    array = []
    yarray = []
    for coin in range(0, 2):

        for ind in [random.randint(0, DataSize - 1)
                    for i in range(int(Num / 2))]:
            array.append(DataDic[coin][ind, :].reshape((size, size, size)))
            yarray.append(coin)

    patch = np.array(array) / 65535
    label = np.array(yarray)

    dic = {x: patch, y_: np.array([Ze[k] for k in yarray]), keep_prob: 1}
    return dic, label


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
DataSize = 500000
path = 'D:/seg/Pdata_b_s/'
# number=1
Ze = {0: [1, 0], 1: [0, 1]}

# tf.reset_default_graph()

batch = 100

x = tf.placeholder("float", shape=[None, size, size, size])
y_ = tf.placeholder("float", shape=[None, 2])
keep_prob = tf.placeholder(tf.float32)

# ------------------------------------------------------
# CNN Construct

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
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
pred = tf.argmax(y, 1)


sess = tf.InteractiveSession()
#merged = tf.summary.merge_all()
tf.global_variables_initializer().run()

saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1000)

# ------------------------------------------------
# Training

for number in range(0, 100):

    # ------------------------------
    # Data csv opened

    with open(path + 'b_' + str(number + 1) + '.csv', 'r') as bdata:
        bdata = bdata.read().splitlines()
    bdata = np.array([[int(j) for j in [bdata[i].split(',')[2:127]][0]]
                      for i in range(len(bdata))])
    print(path + 'b_' + str(number + 1) + '.csv LOADED')

    with open(path + 'o_' + str(number + 1) + '.csv', 'r') as odata:
        odata = odata.read().splitlines()
    odata = np.array([[int(j) for j in [odata[i].split(',')[2:127]][0]]
                      for i in range(len(odata))])
    print(path + 'o_' + str(number + 1) + '.csv LOADED')

    #sdata=np.array([[int(j) for j in [sdata[i].split(',')[2:127]][0]] for i in range(len(sdata))])
    #batch_sy=np.array([Ze[k] for k in [int(sdata[i].split(',')[1]) for i in range(len(sdata))]])

    DataDic = {0: bdata, 1: odata}

    testdic, gt = GenPatchDic_test2(Num=5000)
    print(gt)

	# ------------------------------
	# Load previous models 

    LoadName = 'D:/seg/PatchModels_b/Patch_CNN_Number_' + str(number - 1)
    try:
        saver.restore(sess, LoadName)
        print('Model ' + LoadName + ' successfully loaded')
    except BaseException:
        print('Model ' + LoadName + ' Does not exist, parameters initialized')

	# ---------------------------
	# SGD

    # for i in range(50):
    for cycle in range(10000):
        # print(cycle)
        dic = GenPatchDic(Num=batch, turn=cycle)
        sess.run(train_step, feed_dict=dic)

        if cycle % 5000 == 0:
            print(cycle)
            loss = sess.run(cross_entropy, feed_dict=testdic)
            print(loss)

            print(sess.run(accuracy, feed_dict=testdic))
            Predict = sess.run(pred, feed_dict=testdic)
            print(confusion_matrix(gt, Predict))


	# ----------------------------
	# Save Model

    SaveName = 'D:/seg/PatchModels_b/Patch_CNN_Number_' + str(number)
    save_path = saver.save(sess, SaveName)
    print('Number ' + str(number) + ' training complete, model saved')




