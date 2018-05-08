# -*- coding: utf-8 -*-
"""
Created on Fri May 04 19:45:01 2018

@author: lab
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
from osa_analys import *

routelib = {}
routelib['A','C','E','F'] = 0
routelib['A','C','D','E','F'] = 1
routelib['A','C','D','F']= 2
routelib['A','B', 'E','F']= 3
routelib['A','B','D','E','F']= 4
routelib['A','B','D','E','F']= 4
routelib['A','B','D','F']= 6
#d2
routelib['E','F'] = 0
routelib['D','E','F'] = 1
routelib['D','F'] = 2
#d1
routelib['A'] = 0
routelib['A','C'] = 1
routelib['A','B']=2
routelib['A','B','D']=3
routelib['A','C','D']=4

#Nodes
nodes = ['A','B','C','D','E','F']
nodes2 = ['D','E','F']
nodes1 = ['A','B','C','D']
  
def FindRoute(data):
    global routelib, nodes1
    routeLibKey = []
    for dataKey in nodes1:
        try: 
            temp, dump = osa_analys1(data[dataKey]['amp'], data[dataKey]['wav'])
        except:
            temp = [False]*7
        if temp[6]:  routeLibKey.append(dataKey)
    
    #print routeLibKey
    try:
        return routelib[tuple(routeLibKey)]
    except:
        print routeLibKey #print routeLibKey, '###'
    
def CalPower(data):
    reval = 0
    temp = 10**(data/10)
    return 10*np.log10(sum(temp))
#filedir = "C://Users//Lab//Box Sync//OFC_PD_data//full_qam16//" #change this path
#filedir = "C://Users//lab//Box Sync//OFC_PD_data//d2_qam16//"
filedir = "C://Users//Lab//Box Sync//OFC_PD_data//d1_qam16//"
filelist = os.listdir(filedir)

#Prepare data
outLayer = {}
inLayer = {}

notReadableFile = []
for numI, fn in enumerate(filelist):
    temp = []
    #File  preparation
    f = open(filedir + fn, 'rb')
    data = pickle.load(f)
    f.close()
    #Data processing
    outLayer[numI] = round( data['qt'] )
    
    tempRoute = FindRoute(data)
    if tempRoute == None:
        notReadableFile.append(fn)
        temp.append(-1)

    else: temp.append(tempRoute)
    try: temp.append(data['wss1_att'])
    except: pass
    try: temp.append(data['wss2_att'])
    except: pass
    try: temp.append(data['wss3_att'])
    except: pass

    global nodes1
    for elenodes in nodes1:#2
        try:temp.append( CalPower(data[elenodes]['amp']) )
        except: pass
    inLayer[numI] = temp         
# Statistics of the Q value and routes number relationship.
i = 0
for numI in [-1,0,1,2,3,4]: #None
    temp = []
    for keys in inLayer.keys():
        if  inLayer[keys][0] == numI: #keys < 300 and
            temp.append(outLayer[keys])
            #if temp[-1] > 11: rec_temp = keys
    histdata, bindata =  np.histogram(temp, bins=arange(4,17,1))
    print "Route#", numI, " most Q value:", bindata[argmax(histdata)]
    ploty = histdata#/float(max(histdata))
    #hist(temp, bins = bindata)
    plot(bindata[:-1], ploty, label = 'Route#%d'%numI)

legend()
grid()
xlabel('Q Value')
ylabel('Frequency')

import cPickle as pickle
pickle.dump([inLayer, outLayer], open('ProcessedData1','wb'))
    