from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import cv2
from keras.utils import to_categorical
import matplotlib.pyplot as plt

data = pd.read_csv("./data/train.csv/train.csv")
X_train = []
y_train = []
X_test = []
y_test = []


X_train = np.zeros(shape=(len(data), 48, 48))
y_train = np.array(list(map(int, data['emotion'])))
    
for i, row in enumerate(data.index):
    image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
    image = np.reshape(image, (48, 48))
    X_train[i] = image

X_train = np.array(X_train) / 255.0
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
y_train = np.array(y_train)

num_classes = 7
y_train = to_categorical(y_train, num_classes)

# Create the Sequential model
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
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=128, epochs=50, validation_split=0.1)

# Save the trained model for later use
model.save('2emotion_recognition_model.h5')

