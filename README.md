#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./signdistribution.png "Traffic Sign Distribution (Train, Validation, Test)"
[image2]: ./signsamples.png "Sample Traffic Signs from Training Data"
[image3]: ./signpredictions.png "5 traffic signs downloaded from Internet"

---
###Writeup / README
Here is a link to my [project code](https://github.com/atveit/TrafficSignClassification/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used numpy shape to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? 32x32x3 (3 color channels, RGB)
* The number of unique classes/labels in the data set is ? 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the *normalized* distribution of data for the 43 traffic signs. The key takeaway is that the relative number of data points varies quite a bit between each class, e.g. from around 6.5% (e.g. class 1) to 0.05% (e.g. class 37), i.e. a factor of at least 12 difference (6.5% / 0.05%), this can potentially impact classification performance. 

![alt text][image1]

### 3 Design and Test a Model Architecture

#### 3.1 Preprocessing of images
Did no grayscale conversion or other conversion of train/test/validation images  (they were preprocessed).  For the images from the Internet they were read from using PIL and converted to RGB (from RBGA), resized to 32x32 and converted to numpy array before normalization.

All images were normalized pixels in each color channel (RGB - 3 channels with values between 0 to 255) to be between -0.5 to 0.5 by dividing by (128-value)/255. Did no data augmentation.

Here are sample images from the training set

![alt text][image2]

#### 3.2 Model Architecture

Given the relatively low resolution of Images I started with Lenet example provided in lectures, but to improve training I added Dropout (in early layers) with RELU rectifier functions. Recently read about self-normalizing rectifier function - SELU - so decided to try that instead of RELU. It gave no better end result after many epochs, but trained much faster (got > 90% in one epoch), so kept SELU in the original. For more information about SELU check out the paper [Self-Normalizing Neural Networks](https://arxiv.org/pdf/1706.02515.pdf) from Johannes Kepler University in Linz, Austria. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Dropout     |		keep_prob = 0.9	    |
| SELU	|     |
| Max Pooling				| 2x2 stride, outputs 14x14x6 |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| SELU	|     |
| Dropout     |		keep_prob = 0.9	    |
| Max Pooling				| 2x2 stride, outputs 5x5x16 |
| Flatten | output dimension 400 |
| Fully connected | output dimension 120 |
| SELU |	  |
| Fully connected | output dimension 84 |
| SELU |	  |
| Fully connected | output dimension 84 |
| SELU |	  |
| Fully connected | output dimension 43 |

#### 3.3 Training of Model

To train the model, I used an Adam optimizer with learning rate of 0.002, 20 epochs (converged fast with SELU) and batch size of 256 (ran on GTX 1070 with 8GB GPU RAM)

#### 3.4 Approach to find solution and getting accuracy > 0.93
Adding dropout to Lenet improved test accuracy and SELU improved training speed. The originally partitioned data sets were quite unbalanced (when plotting), so reading all data, shuffling and creating training/validation/test set also helped. I thought about using Keras and fine tuning a pretrained model (e.g. inception 3), but it could be that a big model on such small images could lead to overfitting (not entirely sure about that though), and reducing input size might lead to long training time (looks like fine tuning is best when you have the same input size, but changing the output classes)

My final model results were:
* validation set accuracy of 0.976 (between 0.975-0.982)
* test set accuracy of 0.975

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

Started with Lenet and incrementally added dropout and then several SELU layers.. Also added one fully connected layer more.

* What were some problems with the initial architecture?

No, but not great results before adding dropout (to avoid overfitting)

* Which parameters were tuned? How were they adjusted and why?

Tried several combinations learning rates. Could reduce epochs after
adding SELU. Used same dropout keep rate.

Since the difference between validation accuracy and test accuracy is very low the model seems to be working well. The loss is also quite low (0.02), so little to gain most likely - at least without changing the model a lot.

### 4 Test a Model on New Images

#### 4.1. Choose five German traffic signs found on the web

Here are five German traffic signs that I found on the web:

![alt text][image3]

In the first pick of images I didn't check that the signs actually were among the the 43 classes the model was built for, and that was actually not the case, i.e. making it impossible to classify correctly. But got interesting results (regarding finding similar signs) for the wrongly classified ones, so replaced only 2 of them with sign images that actually was covered in the model, i.e. making it still impossible to classify 3 of them.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compasare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      		| Priority road
| Side road  | Speed limit (50km/h)
| Adult and child on road| Turn left ahead
| Two way traffic ahead| Beware of ice/snow
| Speed limit (60km/h)| Speed limit (60km/h)


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. For the other ones it can`t classify correctly, but the 2nd prediction for sign 3 - "adult and child on road" - is interesting since it suggests "Go straight or right" - which is quite visually similar (if you blur the innermost of each sign you will get almost the same image).



