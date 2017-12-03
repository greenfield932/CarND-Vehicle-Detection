import cv2
import numpy as np
import sys 
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from utilities import *

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
def getWindowsCount(length, start, stop, size, overlap):
    if start == None:
        start = 0
    if stop == None:
        stop = length
    return int(np.floor((stop-start)/size*(1./overlap)-1))

def getStartPos(i, start, size, overlap):
    if start == None:
        start = 0
    return int(start + i*(size*overlap))
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=(None, None), y_start_stop=(None, None), 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    # Compute the span of the region to be searched    
    # Compute the number of pixels per step in x/y
    # Compute the number of windows in x/y
    # Initialize a list to append window positions to
    window_list = []
    w = img.shape[1]
    h = img.shape[0]
    count_w = getWindowsCount(w, x_start_stop[0], x_start_stop[1],xy_window[0], xy_overlap[0])
    count_h = getWindowsCount(h, y_start_stop[0], y_start_stop[1],xy_window[1], xy_overlap[1])
    #print("CountW:"+str(count_w))
    #print("CountH:"+str(count_h))
    
    for i in range(0, count_w):
        for j in range(0, count_h):
            left = getStartPos(i,x_start_stop[0],xy_window[0], xy_overlap[0])
            right = left + xy_window[0]
            top = getStartPos(j,y_start_stop[0],xy_window[1], xy_overlap[1])
            bottom = top + xy_window[1]
            left_top = (left, top)
            right_bottom = (right, bottom)
            window_list.append((left_top, right_bottom))
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        # Calculate each window position
        # Append window position to list
    # Return the list of windows
    return window_list
