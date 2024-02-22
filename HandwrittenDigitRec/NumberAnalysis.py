#handwritten digit recognition based on https://www.youtube.com/watch?v=bte8Er0QhDg 

#[t.r.9K49D] [sou.1SEiW] [ppL.Mh.lN] [m4x.2.wxO] [aU1pq62XV] [P498u5.IX] 
# (03.2.3.01) (05.102.2.66) (48.1.340)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

"""

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

"""

model = tf.keras.models.load_model('handwrittentwo.model')

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This difit is a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("error")
    finally:
        image_number += 1




