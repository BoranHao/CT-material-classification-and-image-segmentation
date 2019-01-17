#!usr/bin/env python
# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------------
'''Shuffle samples for seg_cnn_patch_l2. 100b+100o csvs, 500,000 per file
make sure that each sample is randomly selected from any image and any subtype'''

__author__      = "Boran Hao"
__date__        = "Jan, 2019"
__email__       = "brhao@bu.edu"
#-------------------------------------------------------------------------------


import numpy as np
import random
import copy
import pyfits
import csv



def GenRanMap(filenum=5, samplenum=10):

    ls = [i for i in range(filenum)]
    ds = [ls for i in range(samplenum)]

    for i in range(len(ds)):
        print(i)
        tp = copy.deepcopy(ds[i])
        random.shuffle(tp)
        ds[i] = tp

    ds = np.array(ds).T.tolist()

    for i in range(len(ds)):
        print(i)
        tp = copy.deepcopy(ds[i])
        random.shuffle(tp)
        ds[i] = tp

    return np.array(ds)


def GetDic(d, filenum=5, samplenum=10):

    dic = {}
    for i in range(filenum):
        dic[i] = {}

    for i in range(filenum):
        for j in range(filenum):
            dic[i][j] = []

    # dic=copy.deepcopy(dicc)
    for i in range(filenum):
        print(i)
        for j in range(samplenum):
            dic[i][d[i, j]].append(j)

    return dic


# ---------------------------------------------
# Shuffle according to images

def main():
    ds = GenRanMap(filenum=100, samplenum=500000)
    dic = GetDic(ds, filenum=100, samplenum=500000)


    for i in range(100):
        print(i)
        with open('D:/seg/Pdata_b/b_' + str(i + 1) + '.csv', 'r') as indata:
            indata = indata.read().splitlines()

        for j in range(100):
            print(j)
            if len(dic[i][j]) != 0:
                tardata = open('D:/seg/Pdata_b_s/b_' + str(j + 1) + '.csv', 'a', newline='')
                csv_write = csv.writer(tardata, dialect='excel')

                for ind in dic[i][j]:
                    csv_write.writerow(indata[ind].split(','))


    # ---------------------------------------------
    ds = GenRanMap(filenum=100, samplenum=500000)
    dic = GetDic(ds, filenum=100, samplenum=500000)


    for i in range(100):
        print(i)
        with open('D:/seg/Pdata_b/o_' + str(i + 1) + '.csv', 'r') as indata:
            indata = indata.read().splitlines()

        for j in range(100):
            print(j)
            if len(dic[i][j]) != 0:
                tardata = open('D:/seg/Pdata_b_s/o_' + str(j + 1) + '.csv', 'a', newline='')
                csv_write = csv.writer(tardata, dialect='excel')

                for ind in dic[i][j]:
                    csv_write.writerow(indata[ind].split(','))


    # ----------------------------------------------------
    # Inner shuffle according to samples

    for i in range(100):
        print(i)
        with open('D:/seg/Pdata_b_s/b_' + str(i + 1) + '.csv', 'r') as indata:
            indata = indata.read().splitlines()

        ind = [j for j in range(500000)]
        random.shuffle(ind)
        tardata = open('D:/seg/Pdata_b_s/new/b_' + str(i + 1) + '.csv', 'a', newline='')
        csv_write = csv.writer(tardata, dialect='excel')
        for no in ind:
            # print(j)
            csv_write.writerow(indata[no].split(','))


    for i in range(100):
        print(i)
        with open('D:/seg/Pdata_b_s/o_' + str(i + 1) + '.csv', 'r') as indata:
            indata = indata.read().splitlines()

        ind = [j for j in range(500000)]
        random.shuffle(ind)
        tardata = open('D:/seg/Pdata_b_s/new/o_' + str(i + 1) + '.csv', 'a', newline='')
        csv_write = csv.writer(tardata, dialect='excel')
        for no in ind:
            # print(j)
            csv_write.writerow(indata[no].split(','))


if __name__ == '__main__':
    main()
