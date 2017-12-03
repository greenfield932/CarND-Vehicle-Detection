import cv2
import numpy as np
import sys 
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from utilities import *

def usage():
    print("Usage: pipeline.py video.mp4 [output_video.avi]") 

#main pipeline, frames processed here
def carPipeline(img_orig, debug = False):
    return img_orig

#START

#Video mode or single frame mode
videoMode = True

# if set to True multiple windows with middle processing results will appear
debug = True

#for debug purposes, we can start from some debugging frame
frameStart = 0

#for debugging purposes, in video mode 'a' simbol is waited to move to next frame
oneFrame = False

if videoMode == False:#just load single frame and process it
    
    #img_orig = cv2.imread('test_images/straight_lines1.jpg')
    #img_orig = cv2.imread('test_images/test6.jpg')
    img_orig  = getFrame(sys.argv[1], 0)
    img = carPipeline(img_orig, debug)
    showAndExit(img)
else: #use video for processing

    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    videoWriter = None

    videoFileName = sys.argv[1]
    cap = cv2.VideoCapture(videoFileName)
    
    if cap.isOpened() == False:
    
        print("Error opening video file:" + videoFileName)
        sys.exit(2)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frameCnt = 0

    while(cap.isOpened() and frameCnt < frameStart):
        ret, frame = cap.read()
        frameCnt+=1

    while(cap.isOpened()):
        if cv2.waitKey(25) == 27:
            break

        ret, frame = cap.read()

        #frame = cv2.flip(frame, 0)

        if ret == True: 

            if videoWriter == None and len(sys.argv) >= 3:
                fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
                videoWriter = cv2.VideoWriter(sys.argv[2], fourcc, fps,(frame.shape[1], frame.shape[0]),True)

            frame = carPipeline(frame, debug)
            #cv2.putText(frame, 'Frame: '+str(frameCnt), (20,140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
            #cv2.imshow('Frame',frame)
            showScaled('Output',frame)
            if videoWriter!=None:
                videoWriter.write(frame)
        else: 
            break
        frameCnt+=1

        if oneFrame == True:
            if cv2.waitKey(0) == ord('a'):
                continue
            elif cv2.waitKey(0) == 27:
                break
        
    if videoWriter!=None:
        videoWriter.release()
    cap.release() 
    cv2.destroyAllWindows()
