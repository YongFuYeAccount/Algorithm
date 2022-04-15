# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 22:51:03 2021

@author: 86182
"""
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from PIL import Image 
from sklearn import svm

def img2Array(filename):
    img = Image.open(filename)
    arr=np.asarray(img)
    arr=arr.flatten()
    return arr


#load filename trainingDigigts exist in a list with str
trainingFileList = listdir('trainingDigits')  
hwLabels = []


m = len(trainingFileList)
trainingMat = np.zeros((m,22500))
for i in range(m):
    fileNameStr = trainingFileList[i]
    fileStr = fileNameStr.split('.')[0]
    classNumStr = int(fileStr[0])
    hwLabels.append(classNumStr)
    trainingMat[i,:] = img2Array('trainingDigits/%s' % fileNameStr)
    
model=svm.SVC()
model.fit(trainingMat,hwLabels)  

#iterate through the test set
testFileList = listdir('testDigits')       
errorCount = 0.0
mTest = len(testFileList)
for i in range(mTest):
    fileNameStr = testFileList[i]
    fileStr = fileNameStr.split('.')[0]
    classNumStr = int(fileStr[0])
    vectorUnderTest = img2Array('testDigits/%s' % fileNameStr)
    classifierResult = model.predict(vectorUnderTest.reshape(1,-1))
    print("SVM得到的辨识结果是: %d, 实际值是: %d" % (classifierResult, classNumStr))
    if (classifierResult != classNumStr): errorCount += 1.0
print("\n辨识错误数量为: %d" % errorCount)
print("\n辨识率为: %f ％" % ((1-errorCount/float(mTest))*100))











