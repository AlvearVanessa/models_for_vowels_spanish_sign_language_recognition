"""
Classification Module with FastAI
Based on: Hand Sign Detection course for vowels of the American Sign Language
from Computer Vision Zone course
Websites:
https://www.computervision.zone/courses/hand-sign-detection-asl/
"""

# Import libraries
import numpy as np
import cv2
from fastai.vision.all import *
import fastai
import pathlib

# To handle and manipulate non-Windows file system paths
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


class Classifier:

    def __init__(self, modelPath, labelsPath=None):
        self.model_path = modelPath
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)
        # Load the model
        self.model = load_learner(self.model_path)  #  tensorflow.keras.models.load_model(self.model_path)

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        # self.data = np.ndarray(shape=(1, 400, 400, 3), dtype=np.float32)  #  np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.labels_path = labelsPath
        # Reading the path for labels
        if self.labels_path:
            label_file = open(self.labels_path, "r")
            # Creating a list for labels
            self.list_labels = []
            for line in label_file:
                # The strip method removes any leading spaces at the beginning and trailing the spaces at the end
                stripped_line = line.strip()
                self.list_labels.append(stripped_line)
            label_file.close()
        else:
            print("No Labels Found")

    def getPrediction(self, img, draw=True, pos=(50, 50), scale=2, color=(0, 255, 0)):
        # Run the inference
        prediction = self.model.predict(img)  # self.model.predict(image_batch)
        index = np.argmax(prediction[2]) # ('O', tensor(3), tensor([0.4002, 0.0118, 0.0999, 0.4875, 0.0006]))
        label_index = prediction[0]  # np.argmax(prediction)
        # Drawing the labels
        if draw and self.labels_path:
            cv2.putText(img, label_index,
                        pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)  # str(self.list_labels[index])

        return prediction[0], index  # ('A', tensor(0))



def main():
    cap = cv2.VideoCapture(0)
    maskClassifier = Classifier('C:/Users/maalvear/PycharmProjects/lse_vowels_gr/Model/export_resnet.pkl', 'C:/Users/maalvear/PycharmProjects/lse_vowels_gr/Model/labels.txt')
    while True:
        _, img = cap.read()
        prediction1 = maskClassifier.getPrediction(img)
        print(prediction1)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
