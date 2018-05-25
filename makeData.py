# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 14:03:45 2017

@author: Hannah
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

im = plt.imread('trainingImage.jpg')
#im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
#im = cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
plt.imshow(im)
ctr = plt.ginput(1)
ctrx = int(ctr[0][0])
ctry = int(ctr[0][1])
#cv2.rectangle(im,(ctrx+30,ctry+35),(ctrx-30,ctry-35),(0,0,0),1)
cone = im[ctry-35:ctry+35, ctrx-30:ctrx+30,:]
cv2.imwrite('cone21.jpg',cone)
plt.figure(2)
plt.clf()
plt.imshow(cone)




