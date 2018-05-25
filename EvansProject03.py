# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:13:54 2017

@author: Hannah
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import sklearn
import cv2
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from skimage.feature import hog
import math

with open('SVM.pkl','rb') as a:
    SVM = pickle.load(a)

with open('Scaler.pkl','rb') as b:
    scale = pickle.load(b)
wBGR = []
window=[]
gray = []
pred = []
HOG = []
histH = []
histS = []
feat = []
itr = -1
img = plt.imread('trainingImage.jpg')

im = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h = np.zeros_like(im)
heatMap = h[:,:,0]
step = 6
for i in range(70,im.shape[0]+step,step):
    for j in range(60,im.shape[1]+step,step):
        itr += 1
        wBGR.append(img[i-70:i,j-60:j])
        window.append(im[i-70:i,j-60:j,0])
        gray.append(cv2.cvtColor(wBGR[itr],cv2.COLOR_BGR2GRAY))
        HOG.append(hog(gray[itr],orientations=9,pixels_per_cell=(8,8),
                    cells_per_block=(2,2),visualise=False))
        histH.append(np.histogram(window[itr],bins=20,range=(0,255))[0])
        histS.append(np.histogram(window[itr],bins=20,range=(0,255))[1])
        feat.append(np.hstack((HOG[itr],histH[itr],histS[itr]))) 
scale.fit(feat)
scaleFeat = scale.transform(feat)
pred=(SVM.predict(scaleFeat))

count = -1
for k in range(70,im.shape[0]+step,step):
    for l in range(60,im.shape[1]+step,step):
        count +=1
        if pred[count] == 1:
            heatMap[k-70:k,l-60:l] += 1
            
suppress = heatMap > 2
heatMap[np.where(suppress==False)] = 0
plt.imshow(img)
field = plt.ginput(4)
print('Click the lower left corner of the field first, then continue clockwise')
plt.figure(2)

plt.imshow(heatMap)         
labels = label(heatMap)
plt.figure(3)
plt.clf()
plt.imshow(labels[0])

x = []
y = []
z = []
boxesX = []
boxesY = []
for m in range(labels[1]+1):
    if max(np.where(labels[0]==m)[0]) - min(np.where(labels[0]==m)[0]) > 100:
        #labels[0][np.where(labels[0]==m)] = 0
        print(m, ' not cone')
    elif max(np.where(labels[0]==m)[1]) - min(np.where(labels[0]==m)[1]) > 100:
        #labels[0][np.where(labels[0]==m)] = 0
        print(m,'not cone')
    elif max(np.where(labels[0]==m)[0]) - min(np.where(labels[0]==m)[0]) < 25:
        #labels[0][np.where(labels==m)] = 0
        print(m,'not cone')
    elif max(np.where(labels[0]==m)[1]) - min(np.where(labels[0]==m)[1]) < 25:
        #labels[0][np.where(labels[0]==m)] = 0
        print(m,'not cone')
    else:
        #cv2.rectangle(labels[0],(0,0),(500,500),(255,255,255),4)
        x.append(min(np.where(labels[0]==m)[1]))
        #x.append((max(np.where(labels[0]==m)[1])-min(np.where(labels[0]==m)[1]))/2)
        y.append(max(np.where(labels[0]==m)[0]))
        z.append(1)
        boxesX.append(max(np.where(labels[0]==m)[1]))
        boxesX.append(min(np.where(labels[0]==m)[1]))
        boxesY.append(max(np.where(labels[0]==m)[0]))
        boxesY.append(min(np.where(labels[0]==m)[0]))
        cv2.rectangle(labels[0],(max(np.where(labels[0]==m)[1]),max(np.where(labels[0]==m)[0]))
            ,(min(np.where(labels[0]==m)[1]),min(np.where(labels[0]==m)[0])),
            (255,255,255),3)
        plt.figure(4)
        plt.imshow(img)
        '''
        cv2.rectangle(img,(max(np.where(labels[0]==m)[1]),max(np.where(labels[0]==m)[0]))
            ,(min(np.where(labels[0]==m)[1]),min(np.where(labels[0]==m)[0])),
            (255,255,255),3)
        '''
pt = np.vstack((x,y,z))   
pts = np.float64(pt)
w = np.copy(pts)
box = np.vstack((boxesX,boxesY))   
boxes = np.float64(box)   
plt.figure(5)
plt.clf()   

src = np.float32(field)
dst = np.float32([(0,0),(0,120),(120,120),(120,0)])
M = cv2.getPerspectiveTransform(src,dst)
warp = cv2.warpPerspective(img,M,(120,120))
plt.imshow(warp)
plt.gca().invert_yaxis()
warpPts = np.dot(M,pts)

inside = np.copy(warpPts)
inside[:,np.where(warpPts[0,:]/warpPts[2,:] > 120 )] = np.nan
inside[:,np.where(warpPts[1,:]/warpPts[2,:] > 120)] = np.nan
inside[:,np.where(warpPts[0,:]/warpPts[2,:] < 0)] = np.nan
inside[:,np.where(warpPts[1,:]/warpPts[2,:] < 0)] = np.nan
pts[:,np.where(warpPts[0,:]/warpPts[2,:] > 120)] = np.nan
pts[:,np.where(warpPts[1,:]/warpPts[2,:] > 120)] = np.nan
pts[:,np.where(warpPts[0,:]/warpPts[2,:] < 0)] = np.nan
pts[:,np.where(warpPts[1,:]/warpPts[2,:] < 0)] = np.nan
w[:,np.where(warpPts[0,:]/warpPts[2,:] > 120)] = 0
w[:,np.where(warpPts[1,:]/warpPts[2,:] > 120)] = 0
w[:,np.where(warpPts[0,:]/warpPts[2,:] < 0)] = 0
w[:,np.where(warpPts[1,:]/warpPts[2,:] < 0)] = 0
'''
boxes[:,np.where(warpPts[0,:]/warpPts[2,:] > 120)] = np.nan
boxes[:,np.where(warpPts[1,:]/warpPts[2,:] > 120)] = np.nan
boxes[:,np.where(warpPts[0,:]/warpPts[2,:] < 0)] = np.nan
boxes[:,np.where(warpPts[1,:]/warpPts[2,:] < 0)] = np.nan
'''

plt.plot(inside[0,:]/inside[2,:], inside[1,:]/inside[2,:],'c*')
#,warpPts[1,:]/warpPts[2,:],'c*')

plt.figure(6)
plt.clf()
#pts[np.where(math.isnan(warpPts)] = np.nan
#pts[:,np.where(warpPts[1,:]/warpPts[2,:] < 0)] = np.nan
'''
for n in range(warpPts.shape[1]):
    pts[np.where(math.isnan(warpPts[n][0]))] = np.nan
'''
pts[:,np.where(pts[:,:])]
plt.imshow(img)
xCoord = (warpPts[0,:]/warpPts[2,:])/10
yCoord = (warpPts[1,:]/warpPts[2,:])/10
loop = -1
for o in range(1,len(boxesX),2):
    loop += 1
    if w[0,loop] == 0:
        print(loop,'out of range')
    else:
        cv2.rectangle(img,(boxesX[o-1],boxesY[o]),(boxesX[o],boxesY[o-1]),(255,0,0),3)
        plt.text(pts[0,loop]+2,pts[1,loop]+5,('%.2f'%xCoord[loop],'%.2f'%yCoord[loop]),size=12,color='c')
plt.plot(pts[0,:],pts[1,:],'m*')