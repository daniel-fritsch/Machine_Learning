import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# load model
model = tf.keras.models.load_model('handwrittentwo.model')

# go through the images(samples are provides in digits folder and analyze each individually
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
