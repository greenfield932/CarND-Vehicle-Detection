# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/res_explore1.png
[image2]: ./output_images/colorspace.png
[image3]: ./output_images/Boxes0.jpg
[image4]: ./output_images/Boxes1.jpg
[image5]: ./output_images/Boxes2.jpg
[image6]: ./output_images/Boxes3.jpg
[image7]: ./output_images/sliding_windows.jpg
[image8]: ./output_images/out_0.jpg
[image9]: ./output_images/out_1.jpg
[image10]: ./output_images/out_2.jpg
[image11]: ./output_images/out_3.jpg
[image12]: ./output_images/out_4.jpg
[image13]: ./output_images/out_5.jpg
[image14]: ./output_images/5.jpg
[video1]: ./project_video_out.avi

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `svc_tools.py` function `getFeatures` file lines #19-65.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

My final choice of HOG parameters based on classifier prediction error on a test set and testing multiple configurations for the whole pipeline.
I tried multiple configurations. Good results were obtained with just RGB space 3 channel HOG. YCrCb space provides less false positives, so I picked this space.
YUV looks the same as YCrCb. HLS and LUV looks less stable.
I also tried different orientations count and pixels per cell. I obtained good results with 16 pixels per cell and orentations count 11.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear classifier in `train.py`. I obtained a list of vehicle and non-vehicle images (#39-40), then loaded these images into memory and calculated 
their features #47, #51. Most of code was implemented in `svc_tools.py` and `svc_utils.py` files as separate functions.
Next I split all features on test and train sets (20% of test and 80% of train) using `train_test_split` function from `sklearn.model_selection`,
which shuffles the data as well. Then I created linear classifier (`LinearSVC`) and trained it by `svc.fit(X,Y)` call on line #65.
Line #75 is related to prediction error estimation of the classifier on test set.
I didn't use color features, they didn't help much.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I reused the code provided by udacity as a base and modified it (`svc_common_udacity.py` lines #148-#248). The main idea is to slide a square window among
a region of interest. To speedup the performance HOG features are calculated once for the whole region of interest of the image and then divided by overlapping windows.
Each window used as input for prediction of the classifier.
To find out scales and positions of the windows I added a code that draws all sliding windows on an input image and highlighted active windows with matched prediction.
Using multiple scales and start position I obtained 4 sets of sliding windows (`svc_tools.py` lines #93-95). I re-run my pipeline multiple times and checked what 
window sizes and positions provide more often good predictions.

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched 4 times with two scales and different start positions using YCrCb 3-channel HOG features.
To optimize performance I reduced amount of data for HOG processing by introducing region of interest. The second optimization is to 
compute HOG on the whole region and than apply sliding windows to the computed HOG data.

![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [https://github.com/greenfield932/CarND-Vehicle-Detection/blob/master/project_video_out.avi](./project_video_out.avi)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded 
that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  
I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

To make the algorithm more robust I average heatmap over 6 frames and show boxes that appear minimum on 4 frames. Also I show most recent boxes even no cars were detected up
to 3 frames. It helps to show boxes more stable when a few frames failed to predict cars in a sequence of frames.

Here's an example result showing the heatmap from a series of frames of video, 
the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames and resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image14]


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced a lot of issues with stability of linear classifier as well as combination of HOG parameters and sliding windows. It seems the algorithm is very sensitive to
sliding windows position and sizes. Tuning one parameter may lead to better recognition but may introduce a lot of false positives. Further improvement of the project
can be smart sliding window search, with generating additional windows with different scale and overlapping sizes at places where car was detected 
by the initial grid of windows and using large tresholds for heat maps. It may improve detection stability as well as reduce false positives.
The algorithm will likely fail in a complex environment, like town with multiple objects and buildings (a lot of edges).
I believe deep learning approach with convolutional networks may provide much better results.
