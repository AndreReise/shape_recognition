from keras.datasets import mnist
import cv2
import uuid
import os

(train_X, train_y), (test_X, test_y) = mnist.load_data()

perDigitCount = 200

perDigitFetchedCount = dict()

for i in range(0, len(train_X) - 1):
    currentCount = perDigitFetchedCount.get(train_y[i], 0)

    if(currentCount > perDigitCount):
        continue

    perDigitFetchedCount[train_y[i]] = currentCount + 1

    filename = "{}\\{}\\{}.png".format(os.getcwd(), train_y[i], str(uuid.uuid4()))
    cv2.imwrite(filename, train_X[i])