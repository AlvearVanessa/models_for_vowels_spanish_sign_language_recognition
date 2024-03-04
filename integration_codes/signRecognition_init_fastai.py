"""
LSE Vowels Sign Recognition System Module with FastAI
This is the integration code between the classification and hand-detection modules
Based on: Hand Sign Detection course for vowels of the American Sign Language
from the Computer Vision Zone course
Website:
https://www.computervision.zone/courses/hand-sign-detection-asl/
"""


# Import libraries
import cv2
import HandTrackingModule_noSkeleton as htm
from classificationModule_init_fastai import Classifier
import numpy as np
import math
from fastai.vision.all import *
import fastai
import time

# Capture the webcam
cap = cv2.VideoCapture(0)

# Creating the detector
detector = htm.HandDetector(maxHands=1)
#  Loading trained model from ResNet50_fastai.ipynb
classifier = Classifier("C:/Users/maalvear/PycharmProjects/lse_vowels_gr/Model/export_resnet.pkl", "C:/Users/maalvear/PycharmProjects/lse_vowels_gr/Model/labels.txt")

# Params
offset = 20
imgSize = 400

# Folder to save the images taking in real-time
folder = "C:/Users/maalvear/PycharmProjects/lse_vowels_gr/Data/otros"
counter = 0

# Labels for the vowels
labels = ["A", "E", "I", "O", "U"]


while True:
    # Read the frames from webcam
    success, img = cap.read()
    imgOutput = img.copy()
    # Detecting the hand without skeleton
    hands = detector.findHands(img, draw=False)
    # Crop the image once we have the hand
    if hands:  # If there is something in the hand
        hand = hands[0]  # Because we only have one hand

        # Boundary information
        x, y, w, h = hand['bbox']

        # Creating a white matrix with imgSize x imgSize, with data type np.uint8
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        # The exactly dimensions for crop the images
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]  # initial height  y, final is y+h, initial width x,
        # and, final x+w, the previous step creates our boundary box
        # Put the image crop inside the values on the white matrix
        imgCropShape = imgCrop.shape

        # Relation between height and width
        aspectRatio = h/w

        # Fix the height
        if aspectRatio > 1:
            k = imgSize/h  # Constant for stretching the height
            wCal = math.ceil(k*w)  # wCal is approx the float to the higher integer, e.g. 3.4, round it to 4
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)  # This is the gap to push forward the image in the center
            imgWhite[:,  wGap:wCal + wGap] = imgResize
            # Prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            vowel = labels[index]  # Indicate the vowel recognized
            print(prediction, index)



        # Fix the width
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)  # hCal is approx the float to the higher integer, e.g. 3.4, round it to 4
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)  # This is the gap to push forward the image in the center
            imgWhite[hGap:hCal + hGap, :] = imgResize
            # Prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            # vowel = labels[index]

        # Customize the label to be display
        cv2.putText(imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        # Customize the box for the detecting hand
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)

        # Display
        # cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    # To take a picture while the person is signing
    if key == ord("3"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)





