from flask import Flask, request, redirect, url_for, render_template, Response
import cv2
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from PIL import Image
import matplotlib.pyplot as plt

test_data = pd.read_csv("./data/test.csv/test.csv")
X_train = []
X_train = np.zeros(shape=(len(test_data), 48, 48))
emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}

for i, row in enumerate(test_data.index):
    image = np.fromstring(test_data.loc[row, 'pixels'], dtype=int, sep=' ')
    image = np.reshape(image, (48, 48))
    X_train[i] = image

X_train = np.array(X_train) / 255.0
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

#model
model = Sequential()
# Convolutional Layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the data for Dense layers
model.add(Flatten())

# Dense Layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights('2emotion_recognition_model.h5')

for i in range(4,7):
    plt.imshow(X_train[i], cmap='gray')
    test_image = np.expand_dims(X_train[i], axis = 0)
    result = model.predict(test_image)
    res = np.argmax(result[0])
    print('predicted Label for that image is: {}'.format(emotion_dict[res]))
    plt.show() 

