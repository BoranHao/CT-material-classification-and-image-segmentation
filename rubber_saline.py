#!usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
"""Train SVMs using feature combinations, in order to classify rubber and saline"""

__author__      = "Boran Hao"
__email__       = "brhao@bu.edu"
# -------------------------------------------------------------------------------


import itertools
import csv
import random
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


descriptors = []

for i in range(9):
    ll = list(itertools.combinations([n for n in range(9)], i + 1))
    for j in ll:
        descriptors.append(j)


# -------------------------------------------------------------------------------
# Load data

feature = []
label = []
sfeature = []
tslabel = []

csvFile = open("st2.csv", "r")

reader = csv.reader(csvFile)

for item in reader:
    # print(item)
    label.append(item[0])
    feature.append(item[1:10])


csvFile = open("rt2.csv", "r")

reader = csv.reader(csvFile)

for item in reader:
    # print(item)
    label.append(item[0])
    feature.append(item[1:10])


for ii in feature:
    for j in range(len(ii)):
        ii[j] = float(ii[j])


csvFile = open("sts2.csv", "r")

reader = csv.reader(csvFile)

for item in reader:
    # print(item)
    tslabel.append(item[0])
    sfeature.append(item[1:10])


csvFile = open("rts2.csv", "r")

reader = csv.reader(csvFile)

for item in reader:
    # print(item)
    tslabel.append(item[0])
    sfeature.append(item[1:10])

for ii in sfeature:
    for j in range(len(ii)):
        ii[j] = float(ii[j])


out = open('rs2_cv.csv', 'a', newline='')
csv_write = csv.writer(out, dialect='excel')

ind = [i for i in range(len(label))]
random.seed(2)
random.shuffle(ind)

feature = [feature[i] for i in ind]
label = [label[i] for i in ind]
print(label)


# -------------------------------------------------------------------------------
# Training

def train(fold = 3):

	for comb in descriptors:
		# comb=[0,1,2,3,4,5,6,8]
		# comb=[0]
		# comb=[0,4,5]
		st = ''
		for le in comb:
			st = st + str(le)

		tfeature = []
		tsfeature = []

		for f in feature:
			tfeature.append([f[i] for i in comb])

		for f in sfeature:
			tsfeature.append([f[i] for i in comb])

		fl = int(len(label) / 3)
		flag = [0]
		for nn in range(fold - 1):
			flag.append(fl * (nn + 1))

		# print(flag)

		dic = {}
		dic1 = {}

		for ii in range(-5, 20):
			for jj in range(-15, 20):
				CCR = 0
				tu = (ii, jj)
				ACC = 0
				P1 = 0
				P2 = 0
				Recall = 0
				Spec = 0
				FF1 = 0

				for f in flag:
					ran = [aa for aa in range(f, f + fl)]
					CVTestX = [tfeature[dx] for dx in ran]
					CVTestY = [label[dx] for dx in ran]
					CVTrainX = [tfeature[dd]
								for dd in range(len(label)) if dd not in ran]
					CVTrainY = [label[dd]
								for dd in range(len(label)) if dd not in ran]

					model = SVC(C=2**ii, kernel='rbf', gamma=2**jj)
					model.fit(CVTrainX, CVTrainY)
					pred = model.predict(CVTestX)
					# print(pred)

					cmm = confusion_matrix(CVTestY, pred)
					cmml = cmm.tolist()
					A3 = cmm[0, 0]
					B3 = cmm[0, 1]
					C3 = cmm[1, 0]
					D3 = cmm[1, 1]

					acc = (A3 + D3) / (A3 + B3 + C3 + D3)
					p1 = A3 / (A3 + C3)
					p2 = D3 / (D3 + B3)
					recall = A3 / (A3 + B3)
					spec = D3 / (C3 + D3)
					F1 = 2 * p1 * recall / (p1 + recall)


					ACC += (1 / fold) * acc
					P1 += (1 / fold) * p1
					P2 += (1 / fold) * p2
					Recall += (1 / fold) * recall
					Spec += (1 / fold) * spec
					FF1 += (1 / fold) * F1

				CCR = ACC
				# print(tu)
				# if CCR>=0.8:
				# print(CCR)
				dic[tu] = CCR
				dic1[tu] = [ACC, P1, P2, Recall, Spec, FF1]

		at = max(dic, key=dic.get)
		# print(at)
		# print(dic[at])
		model = SVC(C=2**at[0], kernel='rbf', gamma=2**at[1])
		model.fit(tfeature, label)
		# joblib.dump(model,'cr.m')

		fi = model.predict(tsfeature)
		print(fi)
		print(tslabel)
		cm = confusion_matrix(tslabel, fi)
		print(cm)

		A2 = cm[0, 0]
		B2 = cm[0, 1]
		C2 = cm[1, 0]
		D2 = cm[1, 1]

		print(A2)
		print(B2)
		print(C2)
		print(D2)

		acc = (A2 + D2) / (A2 + B2 + C2 + D2)
		p1 = A2 / (A2 + C2)
		p2 = D2 / (D2 + B2)
		recall = A2 / (A2 + B2)
		spec = D2 / (C2 + D2)
		F1 = 2 * p1 * recall / (p1 + recall)

		cml = cm.tolist()
		lst = dic1[at]

		outp = [comb,at,lst[0],lst[1],lst[2],lst[3],lst[4],lst[5],cml,acc,p1,p2,recall,spec,F1]
		print(outp)
		csv_write.writerow(outp)

		# z=input()


train()

