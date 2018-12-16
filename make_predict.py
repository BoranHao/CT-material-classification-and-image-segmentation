#!usr/bin/env python
# -*- coding: utf-8 -*-


#-------------------------------------------------------------------------------
"""Reload the model and make prediction for patches"""

__author__      = "Boran Hao"
__email__       = "brhao@bu.edu"
#-------------------------------------------------------------------------------


import tensorflow as tf
import random
import csv
import numpy as np
import pyfits
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


Ze = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
LabelDic = {'y': 0, 'b': 1, 's': 2, 'r': 3, 'c': 4}

with open('013s.csv', 'r') as datal:
    data = datal.read().splitlines()

with open('007.csv', 'r') as data2:
    datas = data2.read().splitlines()

print(len(data))
#print([m.split() for m in data[0:2]])
# print([data[0].split(',')[2:67]])
print(np.array([[int(j) for j in [data[i].split(',')[2:67]][0]]
                for i in range(2)]))
print([Ze[k] for k in [LabelDic[[data[i].split(',')[1]][0]] for i in range(2)]])


def RanGen(mini, maxi, no):
    array = []
    while True:
        t = random.randint(mini, maxi)
        if t not in array:
            array.append(t)
            if len(array) == no:
                break
    return array



def main():

    x = tf.placeholder(tf.float32, [None, 64])

    W1 = tf.Variable(tf.truncated_normal([64, 48], stddev=0.1))
    B1 = tf.Variable(tf.zeros([48]))

    W2 = tf.Variable(tf.truncated_normal([48, 32], stddev=0.1))
    B2 = tf.Variable(tf.zeros([32]))

    W3 = tf.Variable(tf.truncated_normal([32, 24], stddev=0.1))
    B3 = tf.Variable(tf.zeros([24]))

    W4 = tf.Variable(tf.truncated_normal([24, 16], stddev=0.1))
    B4 = tf.Variable(tf.zeros([16]))

    W5 = tf.Variable(tf.truncated_normal([16, 8], stddev=0.1))
    B5 = tf.Variable(tf.zeros([8]))

    W6 = tf.Variable(tf.truncated_normal([8, 5], stddev=0.1))
    B6 = tf.Variable(tf.zeros([5]))

    Y1 = tf.nn.selu(tf.matmul(x, W1) + B1)
    Y2 = tf.nn.elu(tf.matmul(Y1, W2) + B2)
    Y3 = tf.nn.selu(tf.matmul(Y2, W3) + B3)
    Y4 = tf.nn.elu(tf.matmul(Y3, W4) + B4)
    Y5 = tf.nn.selu(tf.matmul(Y4, W5) + B5)

    y = tf.nn.softmax(tf.matmul(Y5, W6) + B6)
    y_ = tf.placeholder(tf.float32, [None, 5])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y))
    #cross_entropy = tf.reduce_mean(abs(y_-y))
    train_step = tf.train.GradientDescentOptimizer(
        0.00001).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # print(sess.run(W1))

    pred = tf.argmax(y, 1)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    TestFeedDic = {x: np.array([[int(j) for j in [datas[i].split(',')[2:67]][0]] for i in range(len(datas))]), y_: np.array(
        [Ze[k] for k in [LabelDic[[datas[i].split(',')[1]][0]] for i in range(len(datas))]])}
    TestLabel = np.array([LabelDic[[datas[i].split(',')[1]][0]]
                          for i in range(len(datas))])
    TrainFeedDic = {x: np.array([[int(j) for j in [data[i].split(',')[2:67]][0]] for i in range(len(
        data))]), y_: np.array([Ze[k] for k in [LabelDic[[data[i].split(',')[1]][0]] for i in range(len(data))]])}
    TrainLabel = np.array([LabelDic[[data[i].split(',')[1]][0]]
                           for i in range(len(data))])
    #TestFeature=np.array([[int(j) for j in [data[i].split(',')[2:67]][0]] for i in range(len(data))])
    # print(TestLabel)
    # print(TestFeature)
    #print([LabelDic[[data[i].split(',')[1]][0]] for i in range(len(data))])
    #model = SVC(C=2**at[0], kernel='rbf', gamma=2**at[1])
    '''model = SVC()
	model.fit(TestFeature, TestLabel)
	p=model.predict(TestFeature)
	print(confusion_matrix(TestLabel, p))'''

    batch_x = np.array([[int(j) for j in [data[i].split(',')[2:67]][0]]
                        for i in range(len(data))])
    batch_y = np.array(
        [Ze[k] for k in [LabelDic[[data[i].split(',')[1]][0]] for i in range(len(data))]])
    Total = [i for i in range(len(data))]

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
    saver.restore(sess, 'D:/seg/model')

    Predict = sess.run([pred], feed_dict=TestFeedDic)
    print(confusion_matrix(TestLabel, Predict[0]))
    Predict = Predict[0].tolist()
    # print(Predict)

    out = open('tst.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')

    for l in Predict:
        csv_write.writerow([l])


if __name__ == "__main__":
    with tf.device('/gpu:0'):
        main()
