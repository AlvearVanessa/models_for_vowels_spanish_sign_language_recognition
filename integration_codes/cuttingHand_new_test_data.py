"""
Hand Sign Detection for vowels of the American Sign Language
Based on: Computer Vision Zone
Website: https://www.computervision.zone/courses/hand-sign-detection-asl/
"""

# Import libraries
import cv2
import HandTrackingModule_noSkeleton as htm
import numpy as np
import math
import time
import os

# Capture the webcam
# cap = cv2.VideoCapture(0)

# Creating the detector
detector = htm.HandDetector(maxHands=1)

# Params
offset = 20
imgSize = 400

# Folder to save the images taking in real-time
folder_to_save = "C:/Users/maalvear/PycharmProjects/vowels_lse_gr2/Data/Otros2/U"
counter = 0


# while True:

def list_file_names(origin_path):
    l = []
    for path, subdirs, files in os.walk(origin_path):
        for name in files:
            full_path = os.path.join(path, name)
            l.append(full_path)
    return l  # (os.path.join(path, name))

# C:\Users\maalvear\PycharmProjects\vowels_lse_gr2\new_test_data
new_test_path = "C:/Users/maalvear/PycharmProjects/vowels_lse_gr2/new_test_data/U"
test_list_images = list_file_names(new_test_path)

print(test_list_images)

new_list = [
            'C:/Users/maalvear/PycharmProjects/vowels_lse_gr2/new_test_data/U\\u_person5 (9).jpg'
            ]

for m in new_list:
    print(m)
    # Read the frames from webcam success, img = cap.read()
    # C:\Users\maalvear\PycharmProjects\vowels_lse_gr2\new_test_data\A\a_person1 (1).jpg
    img = cv2.imread(m, cv2.IMREAD_COLOR)
    # Detecting the hand without skeleton
    hands = detector.findHands(img, draw=False)
    # Crop the image once we have the hand
    if hands: # If there is something in the hand
        hand = hands[0]  # Because we only consider one hand

        # Boundary information
        x, y, w, h = hand['bbox']

        # Creating a white matrix with imgSize x imgSize, with data type np.uint8
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        # The exactly dimensions for crop the images
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]  # height initial y, final is y+h, width initial x,
        # final x+w, the previous step creates our boundary box
        # Put the image crop inside the values on the white matrix
        imgCropShape = imgCrop.shape
        #imgWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop # starting points of the height would be 0, the ending point of the heigth is the
        # height of the image crop imgCropShape[0], in this case. For the width imgCropShape[1]

        # Relation between height and width
        aspectRatio = h/w

        # Fix the height
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)  # Approx the float to the higher integer, e.g. 3.4, round it to 4. This wCal is the caluclated
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)  # This is the gap to push forward the image in the center
            imgWhite[:,  wGap:wCal + wGap] = imgResize
            #
            # cv2.imwrite(f'{folder_to_save}/Image_{time.time()}.jpg', imgWhite)
            # cv2.imwrite(os.path.join(folder_to_save, str(k)), imgWhite)
            print(m[67:])
            cv2.imwrite(f'{folder_to_save}/Image_'+ m[67:] +'.jpg', imgWhite)
            #cv2.imwrite(f'{folder_to_save}/Image_{time.time()}.jpg', imgWhite)
            cv2.waitKey(100)


        # Fix the width
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)  # Approx the float to the higher integer, e.g. 3.4, round it to 4. This wCal is the calculated
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)  # This is the gap to push forward the image in the center
            imgWhite[hGap:hCal + hGap, :] = imgResize
            #
            # cv2.imwrite(f'{folder_to_save}/Image_/'+ str(k), imgWhite)
            print(m[67:])
            cv2.imwrite(f'{folder_to_save}/Image_'+ m[67:] +'.jpg', imgWhite)
            cv2.waitKey(100)

        #cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

#    if cv2.imshow("Image", img) == True:
#        counter += 1
#        cv2.imwrite(f'{folder_to_save}/Image_{time.time()}.jpg', imgWhite)
#        print(counter)

















