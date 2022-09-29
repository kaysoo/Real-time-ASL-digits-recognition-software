import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


cap = cv2.VideoCapture(0)  # adding capture object
detector = HandDetector(maxHands=1)  # max hands to track
# load model for prediction
classifier = Classifier("cnn_model4.h5", "labels.txt")

expand = 20
imgSize = 300

labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

# cropping images
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # bounding box information
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-expand:y+h+expand, x-expand:x+w+expand]  # bounding box dimensions for cropping image

        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h  # constant for stretching height
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # resizing image
            # putting cropped image on white image
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)  # calculating width gap for centering image
            imgWhite[0:imgResizeShape[0], wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w  # constant for stretching width
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # resizing image
            # putting cropped image on white image
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)  # calculating height gap for centering image
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # cv2.rectangle(imgOutput, (x - expand, y - expand - 50), (x - expand + 90,y - expand),(255, 0, 255),cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-expand, y-expand), (x+w+expand, y+h+expand), (255, 0, 255), 4)

        # cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)

# webcam real time detection
    cv2.imshow("ASL Detection", imgOutput)
    cv2.waitKey(1)  # one millisecond delay

