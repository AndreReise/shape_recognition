import cv2
import os, shutil
import math
import operator
from scipy.ndimage import interpolation as inter
import numpy as np

def loadData():
    data = dict([
        ('0', []),
        ('1', []),
        ('2', []),
        ('3', []),
        ('4', []),
        ('5', []),
        ('6', []),
        ('7', []),
        ('8', []),
        ('9', []),
    ])

    currentPath = os.getcwd()
    dataDirectory = os.path.join(currentPath, "data")

    for dirPath, _, folderContent in os.walk(dataDirectory):
        for file in folderContent:
            digit = str.split(dirPath, os.path.sep)[-1]
            filePath = os.path.join(dirPath, file)

            if str.split(filePath, '.')[-1] != 'png':
                continue
            
            imageBitmap = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
            _, thresh = cv2.threshold(imageBitmap, 120, 255, cv2.THRESH_BINARY)
            thresh = thresh.astype(int) / 255

            data[digit].append(thresh)

    return data



data = loadData()

def calculateEuclideanDistance(image1, image2):
    if(image1.shape != image2.shape):
        print('Shapes are different')
        return 1000000

    distance = np.sum((image1 - image2) ** 2)
    
    return math.sqrt(distance)

class ClassificationResult:
    recognizedDigit: str
    bitmapIndex: int
    euclideanDistance: float

    def __init__(self, recognizedDigit, bitmapIndex, euclideanDistance):
        self.recognizedDigit = recognizedDigit
        self.bitmapIndex = bitmapIndex
        self.euclideanDistance = euclideanDistance


def classify(preparedImage):
    distances = []

    for digit in data:
        for bitmapIndex, bitmap in enumerate(data[digit]):
            distance = calculateEuclideanDistance(preparedImage, bitmap)
            
            classificationResult = ClassificationResult(digit, bitmapIndex, distance)
            distances.append(classificationResult)
    
    distances.sort(key = lambda result: result.euclideanDistance)

    minDistances = distances[:8]

    frequency = dict()

    for distance in minDistances:
        currentValue = frequency.get(distance.recognizedDigit, 0)
        frequency[distance.recognizedDigit] = currentValue + 1

    max_value = max(frequency.items(), key=operator.itemgetter(1))

    max_value_elements = list()

    for key, value in frequency.items():
        if value == max_value[1]:
            max_value_elements.append(key)

    if len(max_value_elements) == 1:

        for distance in minDistances:
            if distance.recognizedDigit == max_value_elements[0]:
                return distance.recognizedDigit, distance.euclideanDistance, minDistances

    return (minDistances[0].recognizedDigit, minDistances[0].euclideanDistance, minDistances)


# prepare artifactgs directory
artifactsPath = os.path.join(os.curdir, "artifacts")

if os.path.isdir(artifactsPath):
    shutil.rmtree(artifactsPath)
    os.mkdir(artifactsPath)
else:
    os.mkdir(artifactsPath)

sampleName = "sample3.png"
image = cv2.imread(sampleName, cv2.IMREAD_GRAYSCALE)

# In OpenCV, finding contours is like finding white object from black background.
image = cv2.bitwise_not(image)

th, dst = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY);

contours, _ = cv2.findContours(image=dst, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

image_copy =  cv2.imread(sampleName)

count = 1


def findCentroid(image):
    x_acc = 0
    y_acc = 0
    count = 0

    for xi in range(0, image.shape[0] - 1):
        for yi in range(0, image.shape[1]):
            if image[xi, yi] == 255:
                x_acc += xi
                y_acc += yi
                count += 1

    cx = (int)(x_acc / count)
    cy = (int)(y_acc / count)

    return (cx, cy)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    # 1. Select only bounding rectangle
    cropped = image[y:(y + h), x:(x + w), ]

    # 2. Resize to the 20x20 pixel box
    resized = cv2.resize(cropped, (20, 20))

    #cv2.imshow('resized', resized)
    #cv2.waitKey()

    # 3. Extend a little bit to simplify shifting to centroid-based
    extended = cv2.copyMakeBorder(resized, 50, 50, 25, 25, cv2.BORDER_CONSTANT, None, value = 0)

    xc, yc = findCentroid(extended)

    # 4. Shift to centroid baised image
    centroidBased = extended[xc - 14:xc + 14, yc - 14: yc+14]

    # 5. Threshold to True-False shape representation
    _, thresh = cv2.threshold(centroidBased, 120, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(int) / 255

    cv2.imwrite("artifacts/{count}.jpg".format(count = count), centroidBased)

    # 6. Apply classifier
    classificationResult, distance, neighbours = classify(thresh)

    # dump nearest neighbours
    for resultIndex, result in enumerate(neighbours):
        bitmap = data[result.recognizedDigit][result.bitmapIndex]

        #cv2.imshow('result', bitmap)
        cv2.imwrite("artifacts/{count}-{index}.png".format(count = count, index=resultIndex), bitmap, params=[cv2.IMWRITE_PNG_BILEVEL])
        #cv2.waitKey()
    #classificationResult = '1'

    #  Optional - some result writing
    print("Minimal distance for shape {} = {}".format(count, distance))
    cv2.putText(image_copy, classificationResult, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    count = count + 1

cv2.imshow('result', image_copy)
cv2.waitKey()
