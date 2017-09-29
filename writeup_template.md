## Writeup Template

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/Car_NotCar_Examples.png
[image2]: ./examples/Car_Notcar_HogImage.png
[image3]: ./examples/Sliding_windows.png
[image4]: ./examples/HeatMap.png
[image5]: ./examples/sixframes.png
[video1]: ./Project_output_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4th cell of the IPython notebook Vehicle-Detection_P5.ipynb from line 3 and method named get_hog_features. The method extract_features defined in the same code cell was used to extract features of cars and non-car images. extract features method will takes the parameters image, color space,hog paramters and histogram feature flags. This method will call for each image get_hog_features method and concatenates the array of hog features.  

In the first 
I started by reading in all the `vehicle` and `non-vehicle` images. The code for this step is defined in the 3rd code cell of the IPython notebook. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I used this udacity classroom and Q&A video session https://www.youtube.com/watch?v=P2zwrTM8ueA&index=5&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P as a reference to impement the code blocks in this notebook.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed two random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and below are the different combination results

**Combination 2:**

**Paramters:**

color_space = 'RGB'

orient = 9

pix_per_cell = 8

cell_per_block = 2

hog_channel = 'ALL'


**Resutl values:**

82.87636184692383 seconds to compute ....

Feature vector length: 8460

Using: 9 orientations 8 pixels per cell and 2 cells per block

25.78 Seconds to train SVC...

Test Accuracy of SVC =  0.9811

**Combination 2:**

Paramters:

color_space = 'RGB'

orient = 9

pix_per_cell = 8

cell_per_block = 2

hog_channel = 'ALL'

spatial_size = (16,16)


Result values:

79.12372398376465 seconds to compute ....

Feature vector length: 6108

Using: 9 orientations 8 pixels per cell and 2 cells per block

19.48 Seconds to train SVC...

Test Accuracy of SVC =  0.9802

**Combination 3:**

Paramters:

color_space = 'YCrCb'

orient = 9

pix_per_cell = 8

cell_per_block = 2

hog_channel = 'ALL'


**Result values:**

77.53902673721313 seconds to compute ....

Feature vector length: 8460

Using: 9 orientations 8 pixels per cell and 2 cells per block

14.21 Seconds to train SVC...

Test Accuracy of SVC =  0.9917

I selected the 3rd combination from the above combinations as it is giving more accuracy 0.99

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used the udacity image data set contains 8792 car images and 8968 non-car images. The code block is avialable in IPython notebook 6th cell.Trained using SVC classifier for training the model. To extract features method i passed the below paramters.

color_space = 'YCrCb'

orient = 9

pix_per_cell = 8

cell_per_block = 2

hog_channel = 'ALL'

spatial_size = (32,32)

hist_bins = 32

spatial_feat = True

hist_feat = True

hog_feat = True

n_samples = 1000

This method returns the feature vectors of the cars and noncars and then numpy vstack method and then converted to float. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this: 
I used slide_window and search_windows methods to find sliding windows by searching cars. This code is implemented in IPython notebook 7th cell. I experimented with different window sizes like 64X64 and 96X96. I choose 96x96 for window size. I used the HOG feature of the entire image once and then sub sampled that array to extract the features of each and evry window.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./Project_output_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image4]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The major problem i faced to figuring out the a way for filtering the false postives. i tried mutiple combinations by experimenting the various HOG and color space paramters.
The model will not work for real time situations when different vechicles runing on the road like cars, trucks and etc..