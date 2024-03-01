"""
Classification Module with keras
Based on: Teachable Machine, and Hand Sign Detection course for vowels of the American Sign Language
from Computer Vision Zone course
Websites:
https://teachablemachine.withgoogle.com/
https://www.computervision.zone/courses/hand-sign-detection-asl/
"""

# Import libraries
from tensorflow import keras
import numpy as np
import cv2


class Classifier:

    def __init__(self, modelPath, labelsPath=None):
        self.model_path = modelPath
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)
        # Load the model
        self.model = keras.models.load_model(self.model_path)  #  tensorflow.keras.models.load_model(self.model_path)
        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        self.data = np.ndarray(shape=(1, 400, 400, 3), dtype=np.float32)
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
        # Prediction
        # Turn the image into a numpy array
        image_array = np.asarray(img)  # (400, 400, 3)
        # Expand the shape of the array, consider in the first position the number of images in the array
        image_batch = np.expand_dims(image_array, axis=0)   # (1, 400, 400, 3)

        # Run the inference
        prediction = self.model.predict(image_batch)
        index = np.argmax(prediction)

        # Drawing the labels
        if draw and self.labels_path:
            cv2.putText(img, str(self.list_labels[index]),
                        pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

        return prediction, index



def main():
    cap = cv2.VideoCapture(0)
    maskClassifier = Classifier('C:/Users/maalvear/PycharmProjects/lse_vowels_gr/Model/keras_model2', 'C:/Users/maalvear/PycharmProjects/lse_vowels_gr/Model/labels.txt')
    while True:
        _, img = cap.read()
        prediction1 = maskClassifier.getPrediction(img)
        print(prediction1)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
