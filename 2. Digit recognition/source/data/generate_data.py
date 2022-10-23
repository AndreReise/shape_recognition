from keras.datasets import mnist
from matplotlib import pyplot
import cv2
import uuid
import os
import numpy as np
from scipy.ndimage import interpolation as inter

(train_X, train_y), (test_X, test_y) = mnist.load_data()

for i in range(0, len(train_X) - 1):
    filename = "{}\\{}\\{}.png".format(os.getcwd(), train_y[i], str(uuid.uuid4()))
    cv2.imwrite(filename, train_X[i])