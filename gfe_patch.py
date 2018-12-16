#!usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
"""Extract patches in CT images as training data, stored in csv files"""

__author__      = "Boran Hao"
__email__       = "brhao@bu.edu"
# -------------------------------------------------------------------------------


import csv
import numpy as np
import pyfits
import warnings
warnings.filterwarnings("ignore")


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
		self.CurrG = file[0].data
		self.shape = self.CurrG.shape
		print(self.CurrG.shape)

	def GetCT(self, path='I007.fits.gz'):
		f = self.Path + 'TestImages/I' + path + '.fits.gz'
		file = pyfits.open(f, ignore_missing_end=True)
		self.CurrI = file[0].data
		self.CurrFile = path

	def GetSE(self, path='S007.fits.gz'):
		f = self.Path + 'Segmentation/S' + path + '.fits.gz'
		file = pyfits.open(f, ignore_missing_end=True)
		self.CurrS = file[0].data
		self.CurrOut = file
		print(self.CurrS[43, 167, 190])
		# self.Out=copy.deepcopy(self.CurrS)

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
					except:
						pass
			data.close()
		for ls in datal:
			if ls[1] != 0:
				# print(ls[13])
				self.datadic[ls[1]] = ls[13]


		self.Data = datal

	def GetCube(self, size=4):
		Zero = [0 for zz in range(size**3)]
		Count = {'y': 0, 'b': 0, 's': 0, 'r': 0, 'c': 0}
		self.cubesize = size
		for i in range(int(self.shape[0] / self.cubesize)):
		# for i in range(50,51):
			print(i)
			for j in range(int(self.shape[1] / self.cubesize)):
				# print(j)
				for k in range(int(self.shape[2] / self.cubesize)):
					Feature = []
					dic = {'y': 0, 'b': 0, 's': 0, 'r': 0, 'c': 0}
					# print(list(range(4,8)))
					for ii in range(i * self.cubesize, (i + 1) * self.cubesize):
						for jj in range(j * self.cubesize, (j + 1) * self.cubesize):
							for kk in range(k * self.cubesize, (k + 1) * self.cubesize):
								Feature.append(self.CurrI[ii, jj, kk])
								self.GetLabel(ii, jj, kk)
								dic[self.CurrLabel] += 1

								# print(self.CurrLabel)
					if Feature == Zero:
						Count['y'] += 1
						continue
					else:
						Label = max(dic, key=dic.get)
						Count[Label] += 1
						outp = [self.CurrFile, Label]
						outp.extend(Feature)

					csv_write.writerow(outp)

		print(Count)

	def GetLabel(self, x, y, z):
		if self.CurrS[x, y, z] == 0:
			self.CurrLabel = 'y'
			# self.CurrNo=-1
		else:
			if self.CurrG[x, y, z] == 0:
				self.CurrLabel = 'b'
				# self.CurrNo=self.CurrS[x,y,z]
			else:
				if self.datadic[self.CurrG[x, y, z]] in ['s', 'r', 'c']:
					self.CurrLabel = self.datadic[self.CurrG[x, y, z]]
					# self.CurrNo=self.CurrG[x,y,z]
				else:
					self.CurrLabel = 'b'
					# self.CurrNo=self.CurrS[x,y,z]


def main():
	for number in range(13, 14):
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
			ob.GetSE(name)
			ob.GetCT(name)
			ob.GetGT(name)
		except:
			continue

		out = open(name + '.csv', 'a', newline='')
		csv_write = csv.writer(out, dialect='excel')

		ob.GetCSV()
		ob.GetCube()


if __name__ == "__main__":
    main()
