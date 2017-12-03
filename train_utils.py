import glob, os
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import numpy as np
import time
def getFeatures(img):
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2

    imgfeatures = []
    
    for channel in range(img.shape[2]):
        imgfeatures = hog(img[:,:,channel], 
                   orientations=orient, 
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), 
                   transform_sqrt=True, 
                   visualise=False, 
                   feature_vector=True)
    imgfeatures = np.ravel(imgfeatures)    
    return imgfeatures

def getFeaturesBatch(filenames, printProgress = False):
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2

    allfeatures = []
    imgfeatures = []
    currentProgress = 0
    cnt = 0
    size = len(filenames)
    for filename in filenames:
        img = mpimg.imread(filename)
        for channel in range(img.shape[2]):
            imgfeatures = hog(img[:,:,channel], 
                       orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=False, 
                       feature_vector=True)
        imgfeatures = np.ravel(imgfeatures)    
        allfeatures.append(imgfeatures)
        if printProgress==True:
            progress = int(float(cnt+1)/float(size)*100)
            cnt+=1
            if currentProgress != progress:
                currentProgress = progress
                print("progress: "+str(progress)+"%")
    return allfeatures

def findFiles(filter):
    filenames = []
    for filename in glob.iglob(filter, recursive=True):
        filenames.append(filename)
    return filenames

def saveObject(clf, filename):
    joblib.dump(clf, filename) 

def loadObject(filename):
    return joblib.load(filename)

