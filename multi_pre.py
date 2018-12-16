#!usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
"""Load SVMs and network models to make decision"""

__author__      = "Boran Hao"
__email__       = "brhao@bu.edu"
# -------------------------------------------------------------------------------


import warnings
import numpy as np
import csv
import random
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import tensorflow as tf

warnings.filterwarnings("ignore")


# -------------------------------------------------------------------------------
# Load SVMs and features

m2 = joblib.load("cr.m")
m3 = joblib.load("cs.m")
m4 = joblib.load("rs.m")

d2 = [0, 5]
d3 = [0, 1, 2]
d4 = [0, 1, 3, 5, 7, 8]

M = [m2, m3, m4]
D = [d2, d3, d4]

classifier = ['cr', 'cs', 'rs']

nlabel = []
label = []
feature = []


csvFile = open("ts2.csv", "r")
reader = csv.reader(csvFile)

for item in reader:
    # print(item)
    label.append(item[0])
    feature.append(item[1:10])

for ii in feature:
    for j in range(len(ii)):
        ii[j] = float(ii[j])

for k in label:
    if k == 'Back':
        nlabel.append([1, 0])
    else:
        nlabel.append([0, 1])

print(nlabel)


# -------------------------------------------------------------------------------

# Load NN

x = tf.placeholder(tf.float32, [None, 9])

W1 = tf.Variable(tf.truncated_normal([9, 5], stddev=0.1))
B1 = tf.Variable(tf.zeros([5]))

W2 = tf.Variable(tf.truncated_normal([5, 2], stddev=0.1))
B2 = tf.Variable(tf.zeros([2]))

Y1 = tf.nn.relu(tf.matmul(x, W1) + B1)
y = tf.nn.softmax(tf.matmul(Y1, W2) + B2)

y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

prediction = tf.argmax(y, 1)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
# with tf.Session() as sess:
#saver = tf.train.import_meta_graph('/home/ec602/Desktop/NN/bo.meta')
saver.restore(sess, 'D:/VBshare/NN/bo')

# print(sess.run(W1))
fi = sess.run([prediction],feed_dict={x: np.array(feature),y_: np.array(nlabel)})
print(fi)
print(sess.run(accuracy,feed_dict={x: np.array(feature), y_: np.array(nlabel)}))


# -------------------------------------------------------------------------------

pred = []


def make_predict():

    for e in range(len(feature)):
        fe = feature[e]

        if fi[0][e] == 0:
            pred.append('Back')
        else:
            dc = {'c': 0, 's': 0, 'r': 0}
            dcp = {'c': 0, 's': 0, 'r': 0}
            # for sub in typee:
            # dc[sub]=0

            for i in range(3):
                model = M[i]
                modelname = classifier[i]
                sfe = [fe[j] for j in D[i]]
                pp = model.predict([sfe])
                dc[pp[0]] += 1

                prob = model.predict_proba([sfe])
                dcp[modelname[0]] += prob[0][0]
                dcp[modelname[1]] += prob[0][1]

            # print(dcp)
            pre = max(dcp, key=dcp.get)
            pred.append(pre)

    print(pred)

    cm = confusion_matrix(label, pred)
    print(cm)
    print(cm.trace() / len(label))


def main():
    make_predict()


if __name__ == "__main__":
    main()
