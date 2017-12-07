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
from svc_utils import *
from svc_common_udacity import *
from utilities import *

#this function performs feature extraction for all functions in this project
#this code is targeted for education purposes thus I kept all feature extraction 
#parameters here instead of translating them through the whole stack of functions
def getFeatures(image,
                conv='RGB2YCrCb',
                orient=11,
                pix_per_cell=16,
                cell_per_block=2,
                hog_channel='ALL',
                hist_bins = 32,
                hist_range = (0, 256),
                spatial_size = (24, 24),
                useHog = True,
                useSpatial = False,
                useHist = False):
    # apply color conversion if other than 'RGB'
    if conv != None:
        feature_image = convertColor(image,conv)
    else: 
        feature_image = np.copy(image)

    res = ()
    # Call get_hog_features() with vis=False, feature_vec=True
    if useHog == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
            res = res +(hog_features,)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            res = res +(hog_features,)

    # Append the new feature vector to the features list
    if useSpatial == True:
        spatial_features = bin_spatial(image, size=spatial_size)
        res = res +(spatial_features,)

    
    # Apply color_hist() also with a color space option now
    if useHist == True:
        hist_features = color_hist(image, nbins=hist_bins, bins_range=hist_range)
        res = res +(hist_features,)

    # Append the new feature vector to the features list
    return np.concatenate(res)

#this function gets a list of filenames, eeads images from disk and returns a extracts features
#return list of features
def getFeaturesBatch(filenames, 
                    printProgress = False):
    allfeatures = []
    imgfeatures = []
    currentProgress = 0
    cnt = 0
    size = len(filenames)
    for filename in filenames:
        img = mpimg.imread(filename)
        imgfeatures = getFeatures(img)
        allfeatures.append(imgfeatures)
        if printProgress==True:
            progress = int(float(cnt+1)/float(size)*100)
            cnt+=1
            if currentProgress != progress:
                currentProgress = progress
                print("progress: "+str(progress)+"%")
    return allfeatures

#this function performs car search on an image using pretrained svm classifier
#it returns rectangles coordinates where classifier predicts car and a debug image which has pipeline steps on it
def findCars(img, svc, scaler):

    #define windows parameters, region of interst and scale, source image will be resized to that scale prior to window search
    ystarts = [405, 420, 380, 430]#, 415, 380]
    ystops = [520, 520, 560, 645]#, 645, 645]
    scales = [1.0, 1.0, 1.0, 1.4]#, 1.6, 2.0]
    
    draw_img = img.copy()
    total_boxes = []
    colors = getColors()
    totalimg = img.copy()
    for i in range(0, len(scales)):
        ystart = ystarts[i]
        ystop = ystops[i]
        scale = scales[i]
        draw_img = img.copy()
        boxes, draw_img = find_cars(img, draw_img, colors[i], ystart, ystop, scale, svc, scaler, 11, 16, 2, None, None, total_boxes)
        overlay(totalimg,draw_img,0, 50+i*90, 0.5) 
        #showScaled('Boxes'+str(i), cv2.cvtColor(draw_img,cv2.COLOR_RGB2BGR), 0.3)
        #showScaled('Boxes', cv2.cvtColor(totalimg,cv2.COLOR_RGB2BGR), 1.0)
        #showScaled('Boxes', totalimg, 0.5)

    return total_boxes, totalimg
