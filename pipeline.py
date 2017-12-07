import cv2
import numpy as np
import sys 
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from utilities import *
from svc_tools import *
from scipy.ndimage.measurements import label
def usage():
    print("Usage: pipeline.py video.mp4 [output_video.avi]") 

#main pipeline, frames processed here
def carPipeline(img, heat, svc, scaler, frameboxes,heatmap, debug = False):
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    yellow = (255,255,0)
    white=(255,255,255)
    black = (0,0,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    colors = [red, green, blue, white, yellow]
    vehicle_boxes, totalimg = findCars(img, svc, X_scaler)
    averagingFrames = 5 #frames used to establish region
    keepFrames = 10 #how many frames we keep last region if no new vehicle regions appear
                    #this parameter keep regions showing when sometimes we cannot detect vehicles in a couple of frames
    framesWithoutRegions = 0
    useHeat = True
    if useHeat == True:
    # Add heat to each box in box list
        heat2 = np.zeros_like(frame[:,:,0]).astype(np.float)
        heat2 = add_heat(heat2,vehicle_boxes)
        
        if framesWithoutRegions > keepFrames and len(frameboxes)>0:
            framesWithoutRegions = 0
            print("Reset frameboxes")
            for item in frameboxes:
                del item
            heatmap = np.clip(np.zeros_like(frame), 0, 255)
        
        if len(vehicle_boxes) != 0:
            frameboxes.append(heat2)
            framesWithoutRegions = 0
        else:
            framesWithoutRegions+=1
            
        if len(frameboxes)==averagingFrames:
            heatAvg = np.zeros_like(frame[:,:,0]).astype(np.float)
            for h in frameboxes:
                heatAvg += h
            heat = apply_threshold(heatAvg,4)
            del frameboxes[0]
       
            heatmap = np.clip(heat, 0, 255)
        
        hm = np.zeros_like(frame)
        hm[:,:,0] = heat[:]/4*255
        hm[:,:,1] = heat[:]/4*255
        hm[:,:,2] = heat[:]/4*255
        
        
        #showScaled('Heat map',hm, 0.5)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        if debug == True:
            overlay(totalimg, hm, 1000,50, 0.2)
            img = draw_labeled_bboxes(totalimg, labels)
        else:
            img = draw_labeled_bboxes(img, labels)
        
        #showScaled('Heat map',(heat/3*255).astype(np.uint8), 0.5)

        # Apply threshold to help remove false positives
        #heat = apply_threshold(heat,1)

        # Visualize the heatmap when displaying    
        #heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        #labels = label(heatmap)
        #img = draw_labeled_bboxes(img, labels)
        
            
            
    else:
        img = draw_boxes(img, vehicle_boxes, (255,0,0), thick=3)
        
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img, heat, heatmap

#START

#Video mode or single frame mode
videoMode = True

# if set to True multiple windows with middle processing results will appear
debug = True

#for debug purposes, we can start from some debugging frame
frameStart = 0

#for debugging purposes, in video mode 'a' simbol is waited to move to next frame
oneFrame = False

frameboxes = []

vehicleFeaturesFilename = 'vehicle_features.pkl'
nonVehicleFeaturesFilename = 'nonvehicle_features.pkl'
svcModelFileName = 'cardetector.pkl'

svc = loadObject(svcModelFileName)
vehicle_features = loadObject(vehicleFeaturesFilename)
nonvehicle_features = loadObject(nonVehicleFeaturesFilename)
X = np.vstack((vehicle_features, nonvehicle_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)

heat = None
    
heatmap = None
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

            if heat == None:
                heat = np.zeros_like(frame[:,:,0]).astype(np.float)
                heatmap = np.zeros_like(frame[:,:,0]).astype(np.float)
                

            frame, heat, heatmap = carPipeline(frame, heat, svc, X_scaler,frameboxes,heatmap, debug)
            if debug == True:
                cv2.putText(frame, 'Frame: '+str(frameCnt), (20,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
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
