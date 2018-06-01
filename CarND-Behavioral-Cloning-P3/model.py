# Here is the model that I used to train my network

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import skimage.transform as sktransform
import argparse

#----------------------FUNCTIONS-----------------------------------------

# for the augmentation of the data I created 2 functions

def augmentRandBrightnessCameraImages(image):
    imageBr = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    imageBr = np.array(imageBr, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    imageBr[:,:,2] = imageBr[:,:,2]*random_bright
    imageBr[:,:,2][imageBr[:,:,2]>255]  = 255
    imageBr = np.array(imageBr, dtype = np.uint8)
    imageBr = cv2.cvtColor(imageBr,cv2.COLOR_HSV2RGB)
    return imageBr

def addRandomSh(image):
    TopPositionY = 320*np.random.uniform()
    TopPositionX = 0
    BottomPositionX = 160
    BottomPositionY = 320*np.random.uniform()
    imageHls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadowMask = 0*imageHls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadowMask[((X_m-TopPositionX)*(BottomPositionY-TopPositionY) -(BottomPositionX - TopPositionX)*(Y_m-TopPositionY) >=0)]=1
    if np.random.randint(2)==1:
        randomBright = .5
        cond1 = shadowMask==1
        cond0 = shadowMask==0
        if np.random.randint(2)==1:
            imageHls[:,:,1][cond1] = imageHls[:,:,1][cond1]*randomBright
        else:
            imageHls[:,:,1][cond0] = imageHls[:,:,1][cond0]*randomBright
    image = cv2.cvtColor(imageHls,cv2.COLOR_HLS2RGB)
    return image
	
def transImage(image,steer,transRange = 100):
    # Translation
    translateX = transRange*np.random.uniform()-transRange/2
    steeringAngle = steer + translateX/transRange*2*.2
    translateY = 0
    TransformationMatrix = np.float32([[1,0,translateX],[0,1,translateY]])
    ImageTranslated = cv2.warpAffine(image,TransformationMatrix,(320,160))
    return ImageTranslated,steeringAngle

	
#----------------------TAINING DATA PREPERATION-------------------------------------------------------	
	
# Read the csv files and create lines with the paths of the images center left right

lines = []
with open('./training_data_lake/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# starting the preprocesses of the images

images = []
measurements = []
for line in lines:
    correction = 0.25
    
    # we re creating a list with the center images
    sourcePath = line[0]
    filenameCenter = sourcePath.split('/')[-1]
    currentPathCenter = './training_data_lake/data/IMG/' + filenameCenter
    image = cv2.imread(currentPathCenter)
    image = image[55:135, :, :]
    image = cv2.resize(image, (64,64))
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
    # Random change of brightness to the images
    imageBr = augmentRandBrightnessCameraImages(cv2.imread(currentPathCenter))
    imageBr = imageBr[55:135, :, :]
    imageBr = cv2.resize(imageBr, (64,64))
    images.append(imageBr)
    measurements.append(measurement)
    
    # Random change of shadowing to the images
    imageSh = addRandomSh(cv2.imread(currentPathCenter))
    imageSh = imageSh[55:135, :, :]
    imageSh = cv2.resize(imageSh, (64,64))
    images.append(imageSh)
    measurements.append(measurement)
    
    # Use of the left image using correction factor to the steering angle (measurements)
    sourcePathLeft = line[1]
    filenameLeft = sourcePathLeft.split('/')[-1]
    currentPathLeft = './training_data_lake/data/IMG/' + filenameLeft
    imageLeft = cv2.imread(currentPathLeft)
    imageLeft = imageLeft[55:135, :, :]
    imageLeft = cv2.resize(imageLeft, (64,64))
    images.append(imageLeft)
    measurementLeft = measurement + correction
    measurements.append(measurementLeft)
    
    # Use of the right image using correction factor to the steering angle (measurements)
    sourcePathRight = line[2]
    filenameRight = sourcePathRight.split('/')[-1]
    currentPathRight = './training_data_lake/data/IMG/' + filenameRight
    imageRight = cv2.imread(currentPathRight)
    imageRight = imageRight[55:135, :, :]
    imageRight = cv2.resize(imageRight, (64,64))
    images.append(imageRight)
    measurementRight = measurement - correction
    measurements.append(measurementRight)
	
	# Horizontal flipping of the image
    imageFlip = cv2.flip( cv2.imread(currentPathCenter), 1 )
    imageFlip = imageFlip[55:135, :, :]
    imageFlip = cv2.resize(imageFlip, (64,64))
    images.append(imageFlip)
    measurements.append(-measurement)

	# Horizontal translation of the image and adaption of the steering angle
    imageTranslation,steeringTranslation = transImage(cv2.imread(currentPathCenter),measurement)
    imageTranslation = imageTranslation[55:135, :, :]
    imageTranslation = cv2.resize(imageTranslation, (64,64))
    images.append(imageTranslation)
    measurements.append(steeringTranslation)	


X_train = np.array(images)
y_train = np.array(measurements)

#----------------------TAINING DATA PREPERATION-------------------------------------------------------

#import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam

# nVidea Autonomous Car Group model

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 64, 3)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()
model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 8)
model.save('model.h5')
