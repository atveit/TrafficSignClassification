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
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

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

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


