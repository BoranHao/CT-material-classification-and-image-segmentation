#!usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
"""Train the neural network to filter the background"""

__author__      = "Boran Hao"
__email__       = "brhao@bu.edu"
# -------------------------------------------------------------------------------


import random
import csv
import itertools
import warnings
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------
# Load data

feature = []
label = []
sfeature = []
slabel = []


csvFile = open("bt2.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    # print(item)
    label.append([1, 0])
    feature.append(item[1:10])


csvFile = open("ot2.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    # print(item)
    label.append([0, 1])
    feature.append(item[1:10])


for ii in feature:
    for j in range(len(ii)):
        ii[j] = float(ii[j])


csvFile = open("bts2.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    # print(item)
    slabel.append([1, 0])
    sfeature.append(item[1:10])


csvFile = open("ots2.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    # print(item)
    slabel.append([0, 1])
    sfeature.append(item[1:10])

for ii in sfeature:
    for j in range(len(ii)):
        ii[j] = float(ii[j])


ind = [i for i in range(len(label))]
random.seed(2)
random.shuffle(ind)

feature = [feature[i] for i in ind]
label = [label[i] for i in ind]

tf.set_random_seed(5)

# -------------------------------------------------------------------------------

def main():
    x = tf.placeholder(tf.float32, [None, 9])

    W1 = tf.Variable(tf.truncated_normal([9, 5], stddev=0.1))
    B1 = tf.Variable(tf.zeros([5]))

    W2 = tf.Variable(tf.truncated_normal([5, 2], stddev=0.1))
    B2 = tf.Variable(tf.zeros([2]))

    Y1 = tf.nn.relu(tf.matmul(x, W1) + B1)
    y = tf.nn.softmax(tf.matmul(Y1, W2) + B2)

    y_ = tf.placeholder(tf.float32, [None, 2])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    pred = tf.argmax(y, 1)

    xx = []
    yy = []
    yy1 = []

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for _ in range(300000):
        xx.append(_)
        a = random.randint(0, 539)
        while True:
            b = random.randint(0, 539)
            if a != b:
                break
        batch_xs = np.array([feature[a], feature[b]])
        batch_ys = np.array([label[a], label[b]])
        # print(batch_xs)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        yy.append(sess.run(accuracy, feed_dict={x: np.array(sfeature), y_: np.array(slabel)}))
        yy1.append(sess.run(accuracy, feed_dict={x: np.array(feature), y_: np.array(label)}))

    print(sess.run([pred], feed_dict={x: np.array(sfeature), y_: np.array(slabel)}))
    print(sess.run(accuracy, feed_dict={x: np.array(sfeature), y_: np.array(slabel)}))
    save_path = saver.save(sess, 'D:/VBshare/NN/bo')

    plt.figure()

    plt.plot(xx, yy, color='blue', label='test accuracy')
    plt.plot(xx, yy1, color='red', label='training accuracy')

    plt.xlabel("Iteration number")
    plt.ylabel("Accuracy")
    plt.title("Accuracy in iteration steps")
    plt.legend()
    plt.show()


if __name__ == "__main__":
	with tf.device('/gpu:0'):
		main()
