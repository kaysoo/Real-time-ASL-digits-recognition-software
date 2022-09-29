import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)  # adding capture object
detector = HandDetector(maxHands=1)  # max hands to track

expand = 20
imgSize = 300
counter = 0
folder = "Gestures/Images/0"
dataset = "Images"


while True:
    success, img = cap.read()
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

        else:
            k = imgSize / w  # constant for stretching width
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # resizing image
            # putting cropped image on white image
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)  # calculating height gap for centering image
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)  # one millisecond delay
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg', imgWhite)
        print(counter)

