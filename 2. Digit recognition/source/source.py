import cv2
import os
import math
import operator

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

    distance: float = 0

    for x in range(0, image1.shape[0] - 1):
        for y in range(0 , image1.shape[1] - 1):
            pixel1 = image1[x][y]
            pixel2 = image2[x][y]

            if pixel1 != pixel2:
                distance += math.pow((pixel1 - pixel2), 2)
    
    return math.sqrt(distance)

def classify(preparedImage):
    distances = []

    for digit in data:
        for testBitmap in data[digit]:
            distance = calculateEuclideanDistance(preparedImage, testBitmap)

            distances.append([digit, distance])
    
    distances.sort(key = lambda x: x[1])
    minDistances = distances[:8]

    frequency = dict()

    for distance in minDistances:
        currentValue = frequency.get(distance[0], 0)
        frequency[distance[0]] = currentValue + 1

    max_value = max(frequency.items(), key=operator.itemgetter(1))

    max_value_elements = list()

    for key, value in frequency.items():
        if value == max_value:
            max_value_elements.append(key)

    if len(max_value_elements) == 1:
        return max_value_elements[0]
    else:
        return minDistances[0][0]



image = cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)

# In OpenCV, finding contours is like finding white object from black background.
image = cv2.bitwise_not(image)

th, dst = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY);

contours, _ = cv2.findContours(image=dst, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

image_copy =  cv2.imread("sample.png")

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


    #cv2.imshow('cropped', cropped)
    #cv2.waitKey()

    # 2. Resize to the 20x20 pixel box
    resized = cv2.resize(cropped, (20, 20))

    # 3. Extend a little bit to simplify shifting to centroid-based
    extended = cv2.copyMakeBorder(resized, 50, 50, 25, 25, cv2.BORDER_CONSTANT, None, value = 0)

    xc, yc = findCentroid(extended)

    # 4. Shift to centroid baised image
    centroidBased = extended[xc - 14:xc + 14, yc - 14: yc+14]

    # 5. Threshold to True-False shape representation
    _, thresh = cv2.threshold(centroidBased, 120, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(int) / 255

    cv2.imwrite("{count}.jpg".format(count = count), centroidBased)

    # 6. Apply classifier
    classificationResult = classify(thresh)
    #classificationResult = '1'

    #  Optional - some result writing
    cv2.putText(image_copy, classificationResult, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    count = count + 1

cv2.imshow('result', image_copy)
cv2.waitKey()
