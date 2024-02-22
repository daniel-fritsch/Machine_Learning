import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os



#importing the necessary data sets from mnist
mnist = tf.keras.datasets.mnist

#creates tuples to work with the training and testing data from mnist data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#instead of associating brightness with 0-255 we normalize to 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# [the following is the process of defining the model]

#this defines the model we are using from tensorflow and adds the first layer
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

#hidden layers
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

#output layer
model.add(tf.keras.layers.Dense(10, activation='softmax')) #softmax ensures all final outputs add up to 1

#this compiles the model with the adam optimizer function which is a stochastic
#-gradient descent method. Loss function is also defined(not really sure what sparse_categorical_crossentropy 
# means in this context). Metric is used to measure performance
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.save('handwrittentwo.model')
