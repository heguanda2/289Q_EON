#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 2 23:27:37 2018

@author: lab
"""




from __future__ import division
import numpy as np
import sys
import os
import pickle
from osa_analys import *
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

f = open('data_pre_d1_1.txt', 'rb')
data = pickle.load(f)
f.close()

nvalid = 25
niter = 500

class Net(nn.Module):                                         # main frame

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.layer1 = nn.Linear(12, 100, True)
        self.layer2 = nn.Linear(100, 100, True)
        self.layer3 = nn.Linear(100, 250, True)
        self.layer4 = nn.Linear(250, 250, True)
        self.layer5 = nn.Linear(250, 100, True)
        self.layer6 = nn.Linear(100, 30, True)
        self.layer7 = nn.Linear(30, 1, True)
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = self.layer7(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = [Net(), Net(), Net(), Net()]
optimizer = [None] * len(net)
criterion = nn.MSELoss()
lossrec = np.zeros([4, niter])
outputfin = [None] * 4

# train
for ii, neti in enumerate(net):
    neti.cuda()
    optimizeri = optim.Rprop(neti.parameters(), lr=0.001)
    optimizer[ii] = optimizeri
    # wrap them in Variable
    inputstn = torch.FloatTensor(data['nninput'][ii][0:-nvalid]).cuda()     # cast tensors to a CUDA datatype
    labelstn = torch.FloatTensor(data['nnoutput'][ii][0:-nvalid]).cuda()    # cast tensors to a CUDA datatype

    inputs, labels = Variable(inputstn), Variable(labelstn)                 # ???
    # zero the parameter gradients
    for jj in range(niter):
        optimizeri.zero_grad()
    
        # forward + backward + optimize
        outputs = neti(inputs)
        loss = criterion(outputs, labels)                                  # compute loss
        loss.backward()                                                    # back propagate
        optimizeri.step()
        lossrec[ii, jj] = loss.data[0]
    outputfin[ii] = outputs.data.tolist()                                  # output
    plt.plot(lossrec[ii], label='set {}'.format(ii))

plt.legend()
plt.ylim([0, 20])
plt.show()

# valid
lossvalid = [None] * 4
for ii, neti in enumerate(net):
    inputstn = torch.FloatTensor(data['nninput'][ii][-nvalid:]).cuda()
    labelstn = torch.FloatTensor(data['nnoutput'][ii][-nvalid:]).cuda()

    inputs, labels = Variable(inputstn), Variable(labelstn)
    
    outputs = neti(inputs)
    loss = criterion(outputs, labels)
    lossvalid[ii] = loss.data[0]
print(lossvalid)


# full plot
outputfull = []
labelfull = []
for ii, neti in enumerate(net):
    inputstn = torch.FloatTensor(data['nninput'][ii]).cuda()
    #labelstn = torch.FloatTensor(data['nnoutput'][ii]).cuda()

    inputs = Variable(inputstn)
    outputs = neti(inputs)
    #plt.plot(outputs.data.tolist(), labels.data.tolist(), 'b+')
    #plt.show()
    outputfull += outputs.data.tolist()
    labelfull += data['nnoutput'][ii]
    
outputfull = np.array(outputfull).reshape(len(outputfull))
labelfull = np.array(labelfull)
#plt.plot(labelfull, outputfull, '+')
#plt.show()


bs = np.linspace(3, 16, 80)
o, a, b = np.histogram2d(outputfull, labelfull, bs)
a = (a[:-1] + a[1:])/2
b = (b[:-1] + b[1:])/2
plt.figure(figsize = (6,4))
plt.hot()
plt.contourf(a, b, o)
plt.xticks(fontsize = 23)
plt.yticks(fontsize = 23)
plt.rcParams['axes.linewidth']=2.0
plt.show()

# save net
for ii, neti in enumerate(net):
    neti.cpu()
    torch.save(neti, 'net_d1_{}.txt'.format(ii))