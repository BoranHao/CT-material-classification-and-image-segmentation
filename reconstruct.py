#!usr/bin/env python
# -*- coding: utf-8 -*-


#-------------------------------------------------------------------------------
"""Reconstruct the patches into segmentation image"""

__author__      = "Boran Hao"
__email__       = "brhao@bu.edu"
#-------------------------------------------------------------------------------


import csv
import numpy as np
import pyfits


class Rec():

    def __init__(self, filename='007'):
        self.filename = filename
        with open('R' + filename + '.csv', 'r') as datal:
            self.data = datal.read().splitlines()
        # print(datal)

        filer = pyfits.open('D:/seg/Segmentation/S' + filename + '.fits.gz', ignore_missing_end=True)
        # filer[0].data=0
        self.shape = list(filer[0].data.shape)
        print(self.shape)
        self.Mat = np.zeros(self.shape, dtype=int)
        self.cubesize = 4
        self.CurrOut = filer
        self.LabelDic = {'y': 0, 'b': 1, 's': 2, 'r': 3, 'c': 4}

    def Fill(self):
        flag = 0
        for i in range(int(self.shape[0] / self.cubesize)):
            print(i)
            for j in range(int(self.shape[1] / self.cubesize)):
                # print(j)
                for k in range(int(self.shape[2] / self.cubesize)):
                    datal = self.data[flag].split(',')
                    # print(datal)
                    #fe=np.array([int(i) for i in datal[2:69]])
                    #fe=np.array([self.LabelDic[datal[0]] for q in range(self.cubesize**3)])
                    fe = np.array([int(datal[0]) for q in range(self.cubesize**3)])
                    # print(fe)
                    fe = fe.reshape((4, 4, 4))
                    # print(fe)
                    self.Mat[self.cubesize *i:self.cubesize *(i + 1), self.cubesize * j:self.cubesize * (j + 1), self.cubesize * k:self.cubesize * (k + 1)] = fe
                    # print(self.Mat)

                    flag += 1

    def Output(self):
        self.CurrOut[0].data = self.Mat
        self.CurrOut.writeto('O' + self.filename + '.fits.gz')


def main():
	ob = Rec('007')
	ob.Fill()
	ob.Output()


if __name__ == "__main__":
    main()
