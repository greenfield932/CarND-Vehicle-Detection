import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from svc_common_udacity import *

def findFiles(filter):
    filenames = []
    for filename in glob.iglob(filter, recursive=True):
        filenames.append(filename)
    return filenames

def saveObject(clf, filename):
    joblib.dump(clf, filename) 

def loadObject(filename):
    return joblib.load(filename)

def convertColor(image, conv):
    ids = dict()
    ids['RGB2HSV']=cv2.COLOR_RGB2HSV
    ids['RGB2LUV']=cv2.COLOR_RGB2LUV
    ids['RGB2HLS']=cv2.COLOR_RGB2HLS
    ids['RGB2YUV']=cv2.COLOR_RGB2YUV
    
    ids['BGR2HSV']=cv2.COLOR_BGR2HSV
    ids['BGR2LUV']=cv2.COLOR_BGR2LUV
    ids['BGR2HLS']=cv2.COLOR_BGR2HLS
    ids['BGR2YUV']=cv2.COLOR_BGR2YUV
    
    ids['RGB2BGR']=cv2.COLOR_RGB2BGR
    ids['BGR2RGB']=cv2.COLOR_BGR2RGB
    
    ids['RGB2YCrCb']=cv2.COLOR_RGB2YCrCb
    ids['BGR2YCrCb']=cv2.COLOR_BGR2YCrCb
    
    
    if conv in ids:
        return cv2.cvtColor(image, ids[conv])
    print('No acceptable conversions: '+conv)
          
    return image

def getColors():
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    yellow = (255,255,0)
    white=(255,255,255)
    black = (0,0,0)
    aqua = (0,255,255)
    return [aqua, green, blue, white, yellow]
    