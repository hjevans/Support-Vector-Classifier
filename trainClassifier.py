# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:46:25 2017

@author: Hannah
"""
import cv2
import matplotlib.pyplot as plt
import glob
from skimage.feature import hog
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

cones = glob.glob('./cones/cones/*.jpg')
notCones = glob.glob('./cones/notCones/*.jpg')
numCones = 22
numNotCones = 48
numTotal = numCones + numNotCones
data = []
dataHSV = []
feat = [0]*numTotal
HOG = [0]*numTotal
label = []
gray = []
histH = [0]*numTotal
histS = [0]*numTotal

for count in range(numCones):
    imCone = plt.imread(cones[count])
    data.append(imCone)
    label.append(1)

for count in range(numNotCones):
    imNot = plt.imread(notCones[count])
    data.append(imNot)
    label.append(0)

for i in range(numTotal):
    gray.append(cv2.cvtColor(data[i],cv2.COLOR_BGR2GRAY))
    dataHSV.append(cv2.cvtColor(data[i],cv2.COLOR_BGR2HSV))
    HOG[i] = hog(gray[i],orientations=9,pixels_per_cell=(8,8),
                    cells_per_block=(2,2),visualise=False)
    histH[i] = np.histogram(dataHSV[i][:,:,0],bins=20,range=(0,255))[0]
    histS[i] = np.histogram(dataHSV[i][:,:,0],bins=20,range=(0,255))[1]
    feat[i] = np.hstack((HOG[i],histH[i],histS[i]))
scale = StandardScaler()
scale.fit(feat)
scaleFeat = scale.transform(feat)
    
#SVM = sklearn.svm.SVC().fit(scaleFeat,label)

SVM = SVC().fit(scaleFeat,label)

pickle.dump(SVM,open('SVM.pkl','wb'))
pickle.dump(scale,open('Scaler.pkl','wb'))

