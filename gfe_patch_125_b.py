#!usr/bin/env python
# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------------
'''Extract patches and save as csv. 500,000 per file. Choose the large 
background features, and use small prob to choose small features'''

__author__      = "Boran Hao"
__date__        = "Jan, 2019"
__email__       = "brhao@bu.edu"
#-------------------------------------------------------------------------------


import random
import scipy.io
import csv
import numpy as np
import pyfits
import warnings
warnings.filterwarnings("ignore")



# Count={0:0,1:0,2:0,3:0,4:0}
Count = {0: 0, 1: 0}

# Num={0:1,1:1,2:1,3:1,4:1}
Num = {0: 251, 1: 251}

# Type={0:'b_',1:'s_',2:'r_',3:'c_',4:'x_'}
Type = {0: 'b_', 1: 'o_'}

Ze = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}

path = 'D:/seg/Pdata_b/'

out0 = open(path + 'b_' + str(Num[0]) + '.csv', 'a', newline='')
csv_write0 = csv.writer(out0, dialect='excel')

out1 = open(path + 'o_' + str(Num[1]) + '.csv', 'a', newline='')
csv_write1 = csv.writer(out1, dialect='excel')


# out={0:out0,1:out1,2:out2,3:out3,4:out4}
out = {0: out0, 1: out1}
# csv_write={0:csv_write0,1:csv_write1,2:csv_write2,3:csv_write3,4:csv_write4}
csv_write = {0: csv_write0, 1: csv_write1}


def TossCoin(prob):
    coin = random.random()
    if coin < prob:
        re = 1
    else:
        re = 0
    return re


def NewFile(LB):
    Num[LB] += 1
    out[LB].close()
    out[LB] = open(path + Type[LB] + str(Num[LB]) + '.csv', 'a', newline='')
    csv_write[LB] = csv.writer(out[LB], dialect='excel')


def Extract(size=5):
    # -----------------------------------
    for number in range(51, 100):

        if number < 10:
            name = '00' + str(number)
        else:
            if number >= 100:
                name = str(number)
            else:
                name = '0' + str(number)
        print(name)

        Ifile = 'I' + name + '.fits.gz'
        Lfile = 'L' + name + '.fits'
        Sfile = 's' + name + '.fits'

        try:
            FileCT = pyfits.open('D:/seg/TestImages/' + Ifile, ignore_missing_end=True)
            FileCT = FileCT[0].data

            print(FileCT.shape)

            FileLB = pyfits.open('D:/seg/gtl/' + Lfile, ignore_missing_end=True)
            FileLB = FileLB[0].data
            Layer = FileLB.shape[0]
            print(Layer)
        except BaseException:
            continue

        # ----------------------------------
        # Select background samples mianly with large intensity values
        for i in range(int((size - 1) / 2), int(Layer - (size - 1) / 2)):
            print(i)
            for j in range(int((size - 1) / 2), int(512 - (size - 1) / 2)):
                for k in range(int((size - 1) / 2), int(512 - (size - 1) / 2)):
                    if FileLB[i, j, k] != 0 and FileLB[i, j, k] != 4:
                        continue
                        '''label=[Ifile]
						label.append(FileLB[i,j,k])

						feature=FileCT[int(i-(size-1)/2):int(i+(size+1)/2), int(j-(size-1)/2):int(j+(size+1)/2), int(k-(size-1)/2):int(k+(size+1)/2)].reshape((size**3))
						label.extend(feature)
						Count[1]+=1
						#print(label)
						if Count[1]>500000:
							Count[1]=0
							NewFile(1)

						#if label[1]!=0:
						csv_write[1].writerow(label)'''

                    else:
                        if TossCoin(0.05) == 1:
                            label = [Ifile, 0]
                            feature = FileCT[int(i - (size - 1) / 2):int(i + (size + 1) / 2), int(j - (size - 1) / 2):int(
                                j + (size + 1) / 2), int(k - (size - 1) / 2):int(k + (size + 1) / 2)].reshape((size**3))
                            if np.mean(feature) > 50 or np.max(feature) > 800 or TossCoin(0.01) == 1:
                                label.extend(feature)
                                Count[0] += 1
                                # print(label)
                                if Count[0] > 500000:
                                    Count[0] = 0
                                    NewFile(0)

                                # if label[1]!=0:
                                csv_write[0].writerow(label)


def main():
    Extract()


if __name__ == '__main__':
    main()
