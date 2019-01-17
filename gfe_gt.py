#!usr/bin/env python
# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------------
'''Convert the G-images into label values'''

__author__      = "Boran Hao"
__date__        = "Jan, 2019"
__email__       = "brhao@bu.edu"
#-------------------------------------------------------------------------------


import scipy.io
import csv
import numpy as np
import pyfits
import warnings
warnings.filterwarnings("ignore")


TypeNo = {'s': 1, 'r': 2, 'c': 3}
LB = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1]}


class Ext():
    def __init__(self):
        self.Iind = []
        self.CurrI = 0
        self.CurrG = 0
        self.CurrS = 0
        self.Path = 'D:/seg/'

    def GetGT(self, path='G007.fits.gz'):
        f = self.Path + 'gt/G' + path + '.fits.gz'
        file = pyfits.open(f, ignore_missing_end=True)
        self.CurrOut = file
        self.CurrG = file[0].data
        self.shape = self.CurrG.shape
        print(self.CurrG.shape)

    def GetCT(self, path='I007.fits.gz'):
        f = self.Path + 'TestImages/I' + path + '.fits.gz'
        file = pyfits.open(f, ignore_missing_end=True)
        self.CurrI = file[0].data
        self.CurrFile = path

    def GetCSV(self):
        self.datadic = {}
        with open('pdb_v30.csv', 'r') as data:
            datal = data.read().splitlines()
            # print(datal)
            for i in range(len(datal)):
                datal[i] = datal[i].split(',')
                for j in range(len(datal[i])):
                    try:
                        datal[i][j] = int(datal[i][j])
                    except BaseException:
                        pass
            data.close()
        for ls in datal:
            if ls[1] != 0:
                # print(ls[13])
                self.datadic[ls[1]] = ls[13]

        # print(self.datadic)

        self.Data = datal

    def GetCube(self, size=4):
        for i in range(self.shape[0]):
            # for i in range(50,51):
            # print(i)
            for j in range(self.shape[1]):
                # print(j)
                for k in range(self.shape[2]):
                    # print(list(range(4,8)))

                    self.GetLabel(i, j, k)

    def GetLabel(self, x, y, z):
        if self.CurrG[x, y, z] == 0:
            pass
            # self.CurrLabel='b'
            # self.CurrNo=self.CurrS[x,y,z]
        else:
            if self.datadic[self.CurrG[x, y, z]] in ['s', 'r', 'c']:
                self.CurrG[x, y, z] = TypeNo[self.datadic[self.CurrG[x, y, z]]]
                # self.CurrLabel=self.datadic[self.CurrG[x,y,z]]
                # self.CurrNo=self.CurrG[x,y,z]
            else:
                # self.CurrG[x,y,z]=0
                self.CurrG[x, y, z] = 4
                # self.CurrNo=self.CurrS[x,y,z]

    def OutPut(self, path='013'):
        FileLB = self.CurrG
        # FileLB=self.CurrG.tolist()
        '''for i in range(len(FileLB)):
			print(i)
			for j in range(512):
				for k in range(512):
					#FileLB[i][j][k]=LB[FileLB[i][j][k]]
					FileLB[i][j][k]=LB[FileLB[i][j][k]]'''

        # FileLB=np.array(FileLB)
        print(FileLB.shape)
        #scipy.io.savemat('L'+path+'.mat',{'FileLB': FileLB})
        self.CurrOut[0].data = FileLB
        self.CurrOut.writeto('L' + path + '.fits')

def main():
	for number in range(19, 100):
		if number < 10:
			name = '00' + str(number)
		else:
			if number >= 100:
				name = str(number)
			else:
				name = '0' + str(number)

		print(name)
		ob = Ext()

		try:
			ob.GetCT(name)
			ob.GetGT(name)
		except BaseException:
			continue

		ob.GetCSV()
		ob.GetCube()
		ob.OutPut(name)
		
if __name__ == '__main__':
    main()
