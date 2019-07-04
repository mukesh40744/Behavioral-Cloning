# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# **Behavioral Cloning** 

## Writeup Template

###

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


# Relevant Links  i referred below study material for project development 
- A medium article about this project
- - https://medium.com/@mithi/cloning-driving-behavior-with-videogame-like-simulator-and-keras-e31129a8e3b6
- My submitted detailed technical write-up
- - https://github.com/mithi/behavioral-cloning/blob/master/writeup_report.pdf
- The model was largely based from Comma.Ai's code and paper
- - https://github.com/commaai/research/blob/master/train_steering_model.py
- - https://arxiv.org/pdf/1608.01230.pdf
- Check the original project repository
- - https://github.com/udacity/CarND-Behavioral-Cloning-P3
- The simulator that was used can be downloaded here
- - https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip
- The data used to train this model 
- - https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

[//]: # (Image References)

[image1]: ./examples/CNN.PNG 
[image2]: ./examples/taranning and validation set.PNG
[image3]: ./examples/Recovery_from_left.jpg
[image4]: ./examples/Recovery_from_right.jpg


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

For track 1 (Track1 Folder)
	* Track1/model.py containing the script to create and train the model
	* Track1/drive.py for driving the car in autonomous mode
	* Track1/model.h5 containing a trained convolution neural network 
	* track1.mp4
	* track 1 data.csv
	
For track 2 (Track2 Folder)
	* Track2/model.py containing the script to create and train the model
	* Track2/drive.py for driving the car in autonomous mode
	* Track2/model.h5 containing a trained convolution neural network
	* track2.mp4
	*track 2 data.csv
		

* writeup_report.md to summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
For track 1 --   python Track1/drive.py Track1/model.h5

For track 2 --   python Track2/drive.py Track2/model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

  Below is the code for my model which is common for both the track-
  
  def getModel():
    
    model = Sequential()

    model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    
    return model
	
	

only difference is for training the same for track1 I used the Udacity data which is provided at below location-


https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip


for tanning the same model for second track I use my simulator data .

1st I used 7K data . simulator was started working fine but its stopped working after few tracks. so, I provided more data after running the simulator in tanning mode. I rune the simulator in reverse and forward direction to get more data.
 

	
	

 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 70 and 73). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 


for track1 i used the Udacity data which is provided at below location-


https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip


for tanning the same model for second track I use my simulator data .

1st I used 7K data . simulator was started working fine but its stopped working after few tracks. so, I provided more data after running the simulator in tanning mode. I rune the simulator in reverse and forward direction to get more data.
 
 finally, I used more than 20K data for second track.
 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

85% of the samples were used to train while the remaining 15% was used for validation. Each data
sample contains the steering measurement as well as three images captured from three cameras installed at three
different locations in the car [left, center, right]. About 4000 of these samples are within the [-0.05, -0.05] as shown
when plotted in a histogram. Also, the data is biased towards left turns because of the track used to record this
data.

Below method have been used for design this project-
def flipped(image, measurement):
def get_image(i, data):

To combat these biases, we have used the following data augmentation techniques (as seen in the code above):
- For each data point we use, we randomly choose among the three camera positions (left, center, right),
and employ a steering correction of 0.25 for left and .35 for right. The value of steering correction is adjusted based on how
training the network was behaving
- We also randomly flip the image and change the sign of the steering angle.
- Doing those two things above increases our data set by a factor of 6. 
A python generator (as seen in the code below) was used to generate samples for each batch of data that would
generate_samples
when training and validating the network. We used a generator so that we don’t need to store a lot of data unnecessarily and only use the memory that we need to use at a time. Notice that in my code I don’t use the last batch of data which size is less than the batch_size I have chosen. This seems to be not a problem, so I just left it as is.
For this project I have:
- Used a simulator to collect data that could be used as samples of driving behavior
- Built a CNN in keras that predicts steering angles form images
- Trained and validated the model with training and validation set with the data provided by udacity
- Tested that the model could successfully drive around track one without leaving the road

#### 2. Final Model Architecture

def getModel():
    
    model = Sequential()

    model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    
    return model

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 


for track1 I used the Udacity data which is provided at below location-


https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip


attached both the excel sheet for track 1  and track 2

1st i used 7K data . simulator was started working fine but its stopped working after few tracks. so I provided more data after running the simulator in tanning mode. i rune the simulator in reverse and forward direction to get more data.
 
 finally I used more than 20K data for second track.



