# Traffic Sign Recognition WriteUp 

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

## Goals

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Files Submitted

* Writeup: You are reading it!.
* HTML output of the code: [local link](./Traffic_Sign_Classifier.html)
* Project Code: [local link](Traffic_Sign_Classifier.ipynb)

## Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used numpy to compute the required data. The results are the following:

* Total number of examples = 51839
* The size of training set is 34799 (%67.1)
* The size of the validation set is 4410 (%8.5)
* The size of test set is 12630 (%24.4)
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


#### 2. Include an exploratory visualization of the dataset.

For this point, I selected a random image of the set and displayed relevant information. In particular, I wanted to know how the image looked, minimum and maximum pixel values, and related label name (from the signnames.csv file).

I ran manually this code block some number of times, just as a small check to ensure that the images and labels are correct.

### Design and Test a Model Architecture

#### 1. Preprocessing

I just applied a normalization step, to force a zero mean on the input data. First I made sure the source images are encoded as uint8. Then, the normalization is implemted as follows:
* Convert data to float32 type.
* Apply `(data - 128)/128` computation.
* Resulting data will have zero mean and domain [-1,1].

As this preprocessing is just traslation and scaling, and also contains negative numbers, it does not make sense to display resulting images.

#### 2. Model architecture

The final model is the same proposed as a baseline for the project (LeNet 5), but with the following modifications:
* Inputs are RGB images, so 3 channels.
* Output has 43 classes.
* I added 2 dropout layers (with `keep_prob=0.5` for training) after each ReLU on the fully connected layers.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| --------------------- | --------------------------------------------- |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6                  |
| --------------------- | --------------------------------------------- |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16                   |
| --------------------- | --------------------------------------------- |
| Flatten               | input 5x5x16, output 400                      |
| --------------------- | --------------------------------------------- |
| Fully connected		| output 120                                    |
| RELU					|												|
| Dropout               | 												|
| --------------------- | --------------------------------------------- |
| Fully connected		| output 84                                     |
| RELU					|												|
| Dropout               | 												|
| --------------------- | --------------------------------------------- |
| Fully connected		| output 43                                     |
| --------------------- | --------------------------------------------- |
| Softmax				| For Cross-Entropy	computation 				|
| --------------------- | --------------------------------------------- |
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used the baseline `AdamOptimizer`.

The hyperparameters were tunned as follow:
* `BATCH_SIZE = 128`: There was no difference on using bigger (256, 1028) or smaller values, so I kept the default one.
* `EPOCHS = 20`: The optimizer usually reached local minima at around 20-30 steps for every tested configuration.
* `learning rate = 0.001`: The optimizer performed poorly for bigger rates (0.005), while there was no improvement for smaller rates. I prefered keeping the rate as big as possible, for speed issues.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

At first, I tried using the baseline configuration, where the accuracy on the **test set** was around `90%`. Then the normalization preprocessing step increased the accuracy to around `93%`. As a final improvement to address the required results, I modified the baseline arquitecture with 2 dropout layers after the fully connected final layers. The accuracy was further increased to around `95%`. 

As discussed on the lectures, the LeNet arquitecture is already powerfull enough for this problem, just needing some small tweaks. The decision on using the dropout was based on the suggestion from the lectures. The goal was to force the network to generalize over the data.

My final model results were:
* training/validation set accuracy of 95.1%.
* test set accuracy of 93.8%.

 
### Test a Model on New Images

#### 1. Classifying Web Images

Here are five German traffic signs that I found on the web. The images were adjusted, to encode pixels as uint8 and to match the 32x32x3 shape of the dataset examples.

![alt text][web_sample/sample_id7.png]
![alt text][web_sample/sample_id12.png]
![alt text][web_sample/sample_id17.png]
![alt text][web_sample/sample_id18.png]
![alt text][web_sample/sample_id38.png]

The second image might be challenging for the classifier, as some orange borders are removed and there is a bright patch on the left side.

The last image can also be problematic, as the signal perspective is a bit rotated.

#### 2. Predictions for the new Images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (100km/h)	| Speed limit (100km/h)							| 
| Priority Road			| Priotity Road									|
| No entry	      		| No entry                                      |
| General Caution		| General Caution     							|
| Keep right			| Keep right									|

The model was able to correctly classify all the traffic signs, giving an accuracy of 100% over this small test set.

#### 3. How certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

The code for making predictions on my final model is located in the **Output Top 5 Softmax Probabilities For Each Image Found on the Web** section of the IPython notebook.

All the test web images were correctly classified. For each case, the top probability was 1.0. So the classifier was always 100% sure about the classification.

##### ID=7, Label='Speed limit (100km/h)'
| prob | Prediction                                         |
|:----:|:--------------------------------------------------:|
| 1.00 | Speed limit (100km/h)                              |
| 0.00 | Speed limit (80km/h)                               |
| 0.00 | Speed limit (120km/h)                              |
| 0.00 | Speed limit (30km/h)                               |
| 0.00 | Wild animals crossing                              |


##### ID=12, Label='Priority road'
| prob | Prediction                                         |
|:----:|:--------------------------------------------------:|
| 1.00 | Priority road                                      |
| 0.00 | No passing for vehicles over 3.5 metric tons       |
| 0.00 | Traffic signals                                    |
| 0.00 | No entry                                           |
| 0.00 | Right-of-way at the next intersection              |

##### ID=17, Label='No entry'
| prob | Prediction                                         |
|:----:|:--------------------------------------------------:|
| 1.00 | No entry                                           |
| 0.00 | Stop                                               |
| 0.00 | Traffic signals                                    |
| 0.00 | No passing for vehicles over 3.5 metric tons       |
| 0.00 | Priority road                                      |

##### ID=18, Label='General caution'
| prob | Prediction                                         |
|:----:|:--------------------------------------------------:|
| 1.00 | General caution                                    |
| 0.00 | Traffic signals                                    |
| 0.00 | Pedestrians                                        |
| 0.00 | Speed limit (70km/h)                               |
| 0.00 | Bumpy road                                         |

##### ID=38, Label='Keep right'
| prob | Prediction                                         |
|:----:|:--------------------------------------------------:|
| 1.00 | Keep right                                         |
| 0.00 | Turn left ahead                                    |
| 0.00 | Go straight or right                               |
| 0.00 | Road work                                          |
| 0.00 | End of all speed and passing limits                |

