# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains the submission of the Behavioral Cloning project of Self-Driving Car NanoDegree. For this project we will implement a deep convolutional neural network to clone driving behavior. As starting point we used the official [Github repository](https://github.com/udacity/CarND-Behavioral-Cloning-P3) from Udacity. The documentation was written according to the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) and the implementation accomplish with the [rubric points](https://review.udacity.com/#!/rubrics/432/view). 

In order to meet specifications we updated the following files:
* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* this documentation
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

In order to collect data for training and testing our model we used the provided simulator. At the same time, it is necessary in order to perform a video recording of the vehicle driving autonomously around the track. This vehicle needs to perform at least one full lap. The simulator provide us image data from three different cameras and at the same time several measurements such as steering angle which we will want to predict with our model.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies and instructions
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

Rubric Points
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

Moreover we created a jupyter notebook where we test several basic models while following the classroom videos. At the same time, we saved two more basic models for comparison purposes (basic_model.h5 and lenet_model.h5). 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
All the functions are well documented.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

In this section we will describe what our final model architecture looks like (model type, layers, layer sizes, connectivity, etc.) We will include a diagram and/or table describing the final model. The model is inspired on the Nvidia Architecture and in this [Medium post](https://medium.com/udacity/udacity-self-driving-car-nanodegree-project-3-behavioral-cloning-446461b7c7f9) written by Jeremy Shannon. We also tested LeNet model and other basic models with the sample dataset provided by Udacity.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 color image   							| 
| Lambda         		| normalization layer  							| 
| Cropping2D         		| image cropping (we don't want sky info) | 
| Convolution 5x5     	| 2x2 stride, 24 depth,  valid padding, L2(0.001) regularizer	|
| ELU					|				activation function								|
| Convolution 5x5     	| 2x2 stride, 36 depth,  valid padding, L2(0.001) regularizer	|
| ELU					|				activation function								|
| Convolution 5x5     	| 2x2 stride, 48 depth,  valid padding, L2(0.001) regularizer	|
| ELU					|				activation function								|
| Convolution 3x3     	| 2x2 stride, 64 depth,  valid padding, L2(0.001) regularizer	|
| ELU					|				activation function								|
| Convolution 3x3     	| 2x2 stride, 64 depth,  valid padding, L2(0.001) regularizer	|
| ELU					|				activation function								|
| Flatten		|        									|
| Fully connected			| 100, L2(0.001) regularizer    	|
| ELU           |                  |
| Dropout        |  keep_prob= 0.4  |
|	Fully connected			|	50, L2(0.001) regularizer    		|
|	ELU					|												|
| Dropout       |   keep_prob= 0.4  |
| Fully connected | 10, L2(0.001) regularizer    		     |
|	ELU					|												|
| Output | 1    		     |

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on an incremental data set to ensure that the model was not overfitting or underfitting. We followed the provided steps in order to obtain a good data set with enough data. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. In order to avoid overfitting we also used dropout on the fully connected layers.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. We tested with different batch sizes such as 64, 128. The model was trained on a Tesla Xp with enough memory to handle big batch sizes. We also tunned the amount of augmented data to create and the training epochs (with early stopping to stop training properly).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (first track: 3 laps and 3 laps in te opposite direction. second track: 1 lap). I also record data when recovering from the left and right sides of the road to he center. Moreover, I also used the provided data by Udacity and also perform data augmentation by flipping images and inverting steering values.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to improve a basic network and incrementally add more data in order to avoid overfitting. Following the nVidia architecture and other usefull tips we were able to achieve a good model with a good performance.

My first step was to train a model with a few convolutions and with only the sample data provided by Udacity. Then I added the two preprocessing layers (Lambda and Cropping2D) in order to normalize data and avoid useless information from images. Results improved, nevertheless car wasn't driving properly. Due to, we added more layers getting LeNet model architecture. This was a very good improvement, nevertheless the data wasn't enough in order to 
ensure that there wasn't overfitting. 

In order to deal with the lack of data, we record more data with the simulator. Also, we used left and right cameras and left and right measurements applying a correction value. Moreover, we do data augmentation flipping the images and inverting steering measure. The car was driving more accurately but loosing the control. The driving wasn't smooth at all. 

We also tested preprocessing the image changing the color space. Results in our case didn't improve significantly.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model adding Dropout layers.

The final improvement was using the architecture from nVidia changing some details such as:
* adding dropout layers
* adding L2 regularization
* using ELU activation function

The final step was to run the simulator to see how well the car was driving around track one. We can see the result in the provided video. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving. 

I then recorded the three laps on track in the opposite direction using center lane driving as well. 

I also recorded one lap on the second and more difficult track.

Finally, I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to deal with difficult situations.

To augment the data sat, I also flipped images and angles simulating driving in the opposite direction.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was  as evidenced by early stopping callback. I used an adam optimizer so that manually training the learning rate wasn't necessary.
