#!usr/bin/env python
# -*- coding: utf-8 -*-


#-------------------------------------------------------------------------------
"""Extract features using groundtruth labels and CT images"""

__author__      = "Boran Hao"
__email__       = "brhao@bu.edu"
#-------------------------------------------------------------------------------


import sys
import os
import re
import csv
import scipy.io as sio
from numpy import linalg as la
import random
import copy
from shutil import move, copy
import pickle
import pyfits
import math
import matplotlib.pyplot as plt
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# print(round((632-676)/10))
# direction=[(0,0,1),(0,1,0),(1,0,0),(0,1,1),(0,-1,1),(1,1,0),(-1,1,0),(1,0,1),(-1,0,1),(1,1,1),(-1,1,1),(-1,-1,1),(1,-1,1)]


class Ext():
    def __init__(self):
        self.Iind = []
        self.CurrI = 0
        self.CurrG = 0
        self.CurrS = 0

    def GetGT(self, path='G007.fits.gz'):
        file = pyfits.open(path, ignore_missing_end=True)
        self.CurrG = file[0].data
        # print(self.CurrG)

    def GetCT(self, path='I007.fits.gz'):
        file = pyfits.open(path, ignore_missing_end=True)
        self.CurrI = file[0].data
        self.CurrFile = path

    def GetSE(self, path='S007.fits.gz'):
        file = pyfits.open(path, ignore_missing_end=True)
        self.CurrS = file[0].data
        self.CurrOut = file
        # print(self.CurrS[43,167,190])
        # self.Out=copy.deepcopy(self.CurrS)

    def GetLabel(self):
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
        print(datal[15][14])
        self.Data = datal

    def GetMat(self):
        a = ob.CurrG.nonzero()
        index = np.asarray(a).T
        print(index.shape)
        b = self.CurrG[index[:, 0], index[:, 1], index[:, 2]]
        c = self.CurrI[index[:, 0], index[:, 1], index[:, 2]]
        print(b.shape)
        print(c.shape)
        d = np.vstack((index.T, b.T, c.T)).T
        print(d)
        self.Mat = d

    def GetFeat(self):
        a = self.Mat[:, 3].tolist()
        # print(a)
        ob_no = set(a)
        print(self.CurrFile)

        print(ob_no)
        for no in ob_no:
            print(' ')
            print(no)

            for ls in self.Data:
                if ls[1] == no:
                    tp = ls[14]
                    break
            try:
                print(tp)
            except UnboundLocalError:
                csv_write.writerow([self.CurrFile, no, 'GT Data Lost'])
                continue

            if tp != 'pt' and tp != 'l':
                idx = [i for i in range(len(a)) if a[i] == no]
                ind = np.array([self.Mat[i, 4]
                                for i in range(len(a)) if a[i] == no]).T
                # print(ind)
                meann = ind.mean()
                m3 = (ind**3).mean()
                varr = ind.var()
                stdd = varr**0.5

                k3 = (ind - meann)**3
                sk = k3.mean() / stdd**3

                k4 = (ind - meann)**4
                ku = k4.mean() / stdd**4 - 3

                print(meann)
                print(varr)
                print(sk)
                print(ku)

                direction = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (0, -1, 1), (1, 1, 0),
                             (-1, 1, 0), (1, 0, 1), (-1, 0, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1)]
                # direction=[(0,0,1),(0,1,0)]

                MP = 0
                EN = 0
                ASM = 0
                CON = 0
                HOM = 0

                for d in direction:
                    # dhist=np.zeros([1,4096],dtype=int)
                    dhist = {}

                    for n in idx:
                        try:
                            ina = int(
                                self.CurrI[self.Mat[n, 0], self.Mat[n, 1], self.Mat[n, 2]])
                            inb = int(
                                self.CurrI[self.Mat[n, 0] + d[0], self.Mat[n, 1] + d[1], self.Mat[n, 2] + d[2]])

                            if inb != 0:
                                # print(ina)
                                # print(inb)
                                dif = int((ina - inb))
                                try:
                                    dhist[dif] += 1
                                except KeyError:
                                    dhist[dif] = 0
                                    dhist[dif] += 1
                        except IndexError:
                            pass

                    # print(dhist)

                    lsl = dhist.values()
                    # lsl=np.array(lsl)
                    # lsl=lsl[0]
                    he = sum(lsl)

                    for key in dhist.keys():
                        dhist[key] = dhist[key] / he

                    # print(sum(dhist.values()))

                    # print(dhist)
                    tMP = max(dhist.values())
                    # print(tMP)

                    tEN = 0
                    tASM = 0
                    tCON = 0
                    tHOM = 0
                    for key in dhist.keys():
                        # if dhist[ii]!=0:
                        p = dhist[key]
                        v = key

                        tEN += p * math.log(p)
                        tASM += p**2
                        tCON += (v**2) * p
                        tHOM += p / (1 + abs(v))

                    MP += (1 / 13) * tMP
                    EN += (1 / 13) * tEN
                    ASM += (1 / 13) * tASM
                    CON += (1 / 13) * tCON
                    HOM += (1 / 13) * tHOM

                print(MP)
                print(EN)
                print(ASM)
                print(CON)
                print(HOM)

                outp = [self.CurrFile,no,tp,meann,varr,sk,ku,MP,EN,ASM,CON,HOM]
                csv_write.writerow(outp)
                #plt.bar(range(len(dhist)), dhist,fc='r')
                # plt.show()

    def GetSMat(self):
        index = ob.CurrS.nonzero()
        index = np.asarray(index).T
        print(index.shape)
        b = self.CurrS[index[:, 0], index[:, 1], index[:, 2]]
        c = self.CurrI[index[:, 0], index[:, 1], index[:, 2]]
        print(b.shape)
        print(c.shape)
        d = np.vstack((index.T, b.T, c.T)).T
        print(d)
        self.SMat = d

    def GetSFeat(self):
        a = self.SMat[:, 3].tolist()
        # print(a)
        ob_no = set(a)
        print(self.CurrFile)

        print(ob_no)
        for no in ob_no:
            #print(' ')
            # print(no)
            '''for ls in self.Data:
                    if ls[1]==no:
                            tp=ls[14]
                            break
            if tp!='pt' and tp!='l':'''
            idx = [i for i in range(len(a)) if a[i] == no]
            bl = int(0.7 * len(idx))
            cun = 0
            for n in idx:
                if self.CurrG[self.SMat[n, 0],
                              self.SMat[n, 1], self.SMat[n, 2]] == 0:
                    cun += 1
            if cun > bl:
                tp = 'Back'
                print(' ')
                print(no)
                print(tp)

                ind = np.array([self.SMat[i, 4]
                                for i in range(len(a)) if a[i] == no]).T
                # print(ind)
                meann = ind.mean()
                m3 = (ind**3).mean()
                varr = ind.var()
                stdd = varr**0.5

                k3 = (ind - meann)**3
                sk = k3.mean() / stdd**3

                k4 = (ind - meann)**4
                ku = k4.mean() / stdd**4 - 3

                print(meann)
                print(varr)
                print(sk)
                print(ku)

                # dhist=np.zeros([1,2500],dtype=int)

                # direction=[(0,0,1),(0,1,0)]
                direction = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (0, -1, 1), (1, 1, 0),
                             (-1, 1, 0), (1, 0, 1), (-1, 0, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1)]

                MP = 0
                EN = 0
                ASM = 0
                CON = 0
                HOM = 0

                for d in direction:
                    # dhist=np.zeros([1,4096],dtype=int)
                    dhist = {}

                    for n in idx:
                        try:
                            ina = int(
                                self.CurrI[self.SMat[n, 0], self.SMat[n, 1], self.SMat[n, 2]])
                            inb = int(
                                self.CurrI[self.SMat[n, 0] + d[0], self.SMat[n, 1] + d[1], self.SMat[n, 2] + d[2]])

                            if inb != 0:
                                # print(ina)
                                # print(inb)
                                dif = int((ina - inb))
                                try:
                                    dhist[dif] += 1
                                except KeyError:
                                    dhist[dif] = 0
                                    dhist[dif] += 1
                        except IndexError:
                            pass

                    # print(dhist)

                    lsl = dhist.values()
                    # lsl=np.array(lsl)
                    # lsl=lsl[0]
                    he = sum(lsl)

                    for key in dhist.keys():
                        dhist[key] = dhist[key] / he

                    # print(sum(dhist.values()))

                    # print(dhist)
                    tMP = max(dhist.values())
                    # print(tMP)

                    tEN = 0
                    tASM = 0
                    tCON = 0
                    tHOM = 0
                    for key in dhist.keys():
                        # if dhist[ii]!=0:
                        p = dhist[key]
                        v = key

                        tEN += p * math.log(p)
                        tASM += p**2
                        tCON += (v**2) * p
                        tHOM += p / (1 + abs(v))

                    MP += (1 / 13) * tMP
                    EN += (1 / 13) * tEN
                    ASM += (1 / 13) * tASM
                    CON += (1 / 13) * tCON
                    HOM += (1 / 13) * tHOM

                print(MP)
                print(EN)
                print(ASM)
                print(CON)
                print(HOM)

                outp = [self.CurrFile,no,tp,meann,varr,sk,ku,MP,EN,ASM,CON,HOM]
                csv_write.writerow(outp)

                #plt.bar(range(len(dhist)), dhist,fc='r')
                # plt.show()

    def GetSHist(self):
        self.CurrTar = []
        a = self.SMat[:, 3].tolist()
        # print(a)
        ob_no = set(a)
        print(ob_no)
        self.CurrObs = list(ob_no)
        segments = []
        for no in ob_no:
            ind = np.array([i for i in range(len(a)) if a[i] == no]).T
            #ran=random.shuffle([ii for ii in range(int(0.1*ind.shape[0]))])
            ran = [i for i in range(ind.shape[0])]
            sh = [ind[i] for i in ran]
            typp = {}
            for idd in sh:
                typ = self.CurrG[self.SMat[idd, 0],
                                 self.SMat[idd, 1], self.SMat[idd, 2]]
                if typ != 0:
                    if typ in typp.keys():
                        typp[typ] += 1
                    else:
                        typp[typ] = 0
                        typp[typ] += 1
            print(typp)
            if typp == {}:
                typee = []
            else:
                typee = max(typp, key=typp.get)
                for ls in self.Data:
                    if ls[1] == typee:
                        typee = ls[14]
                        break
            print(ind.shape)
            obj = self.SMat[ind, 4]
            hist = np.histogram(obj, bins=range(0, 4500, 10), normed=True)[0]
            # print(obj.shape)

            pre = {'s': 0, 'r': 0, 'l': 0, 'c': 0}
            fea = copy.deepcopy(hist.tolist())
            #del fea[448]
            for mo in model:
                p = mo.predict(fea)
                pre[p[0]] = pre[p[0]] + 1
            pred = max(pre, key=pre.get)
            print(no)
            print(pred)
            print(typee)

            if pred == 's':
                self.CurrTar.append(no)

            dic = {'No': no, 'GTType': typee, 'Hist': hist}
            # print(dic)
            segments.append(dic)
        self.segment = segments


# nolist=['07','09']
nolist = ['81','82','83','84','85','86','87','88','89','90','91','92','93']

out = open('kong.csv', 'a', newline='')
csv_write = csv.writer(out, dialect='excel')


def main():
    for no in nolist:
        print(no)

        # Gfile='G0'+no+'.fits.gz'
        # Ifile='I0'+no+'.fits.gz'
        # Sfile='S0'+no+'.fits.gz'

        Gfile = 'G1' + no + '.fits.gz'
        Ifile = 'I1' + no + '.fits.gz'
        Sfile = 'S1' + no + '.fits.gz'

        ob = Ext()
        ob.GetSE(Sfile)
        ob.GetCT(Ifile)
        ob.GetGT(Gfile)

        ob.GetLabel()

        ob.GetMat()
        ob.GetFeat()

        ob.GetSMat()
        ob.GetSFeat()


if __name__ == "__main__":
    main()
