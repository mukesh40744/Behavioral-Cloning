'''
Created on Feb 22, 2019

@author: SG0301464
'''
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Lambda, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
from scipy.misc import imread
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random


def flipped(image, measurement):
    return np.fliplr(image), -measurement


def get_image(i, data):
    
    positions, corrections = ['left', 'center', 'right'], [.25, 0, -.35]
    ID, r = data.index[i], random.choice([0, 1, 2])

    measurement = data['steering'][ID] + corrections[r]

    path = PATH + data[positions[r]][ID]
    if r == 1: path = PATH + data[positions[r]][ID]
    image = imread(path)

    if random.random() > 0.5:
        image, measurement = flipped(image, measurement)

    return image, measurement


def generate_samples(data, batch_size):

  while True:

    SIZE = len(data)
    data.sample(frac=1)

    for start in range(0, SIZE, batch_size):
      images, measurements, speeds = [], [], []

      for this_id in range(start, start + batch_size):
        if this_id < SIZE:
          image, measurement, speed = get_image(this_id, data)
          measurements.append(measurement)    
          speeds.append((speed))
          images.append(image)

      yield np.array(images), np.array(measurements), np.array(speeds)


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


model = getModel()

model.summary()
model.compile(optimizer="adam", loss="mse")

BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 10
PATH = "data/"
CSV_FILE = "driving_log.csv"

DATA = pd.read_csv(PATH + CSV_FILE, usecols=[0, 1, 2, 3])

TRAINING_DATA, VALIDATION_DATA = train_test_split(DATA, test_size=0.15)
TOTAL_TRAIN_DATA = len(TRAINING_DATA)
TOTAL_VALID_DATA = len(VALIDATION_DATA)


training_generator = generate_samples(TRAINING_DATA, batch_size=BATCH_SIZE)
validation_generator = generate_samples(VALIDATION_DATA, batch_size=BATCH_SIZE)

history_object = model.fit_generator(training_generator,
                 samples_per_epoch=TOTAL_TRAIN_DATA,
                 validation_data=validation_generator,
                 nb_val_samples=TOTAL_VALID_DATA,
                 nb_epoch=NUMBER_OF_EPOCHS,
                 verbose=1)

print('Saving model...')
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
model.save("model.h5")

with open("model.json", "w") as json_file:
  json_file.write(model.to_json())

print("Model Saved.")
