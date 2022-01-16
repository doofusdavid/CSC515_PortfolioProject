'''
Module 8 - Portfolio Project
David Edwards
CSC515 - Foundations of Computer Vision
'''
import numpy as np
import cv2
import math
import dlib
import os


def draw_found_faces(detected, image, color: tuple):
    for (x, y, width, height) in detected:
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            color,
            thickness=2
        )


def detect(img):
    face_cascade = cv2.CascadeClassifier(
        "/usr/local/Cellar/opencv/4.5.3_3/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
    eyes_cascade = cv2.CascadeClassifier(
        "/usr/local/Cellar/opencv/4.5.3_3/share/opencv4/haarcascades/haarcascade_eye.xml")
    img = cv2.imread(img, 1)
    (height, width) = img.shape[:2]
    blur_kernel = int(height/10)
    blurred_image = cv2.blur(img, (blur_kernel, blur_kernel), 0)
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    eye_size = int(height/500)
    # converting to gray image for faster video processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(
        130, 130), flags=cv2.CASCADE_SCALE_IMAGE)
    # if at least 1 face detected
    if len(faces) >= 0:
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            eyes = eyes_cascade.detectMultiScale(gray[y:y + h, x:x + w], scaleFactor=1.4, minNeighbors=10, minSize=(
                50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

            if len(eyes) < 2:
                print("Wrong number of eyes - expected 2, got " + str(len(eyes)))
                continue
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 105), 2)

                roi_mask = mask[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eye_count = 0
                for (ex, ey, ew, eh) in eyes:
                    eye_count += 1
                    if eye_count > 2:
                        continue
                    cv2.rectangle(roi_mask, (ex, ey),
                                  (ex+ew, ey+eh), (255, 255, 255), -1)
        out = np.where(mask == np.array(
            [255, 255, 255]), blurred_image, img)
        cv2.imshow('Face Detection', mask)
        # wait for 'c' to close the application
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_image(img):
    """
    Show image (for debugging)
    """
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
    Iterate through img directory and split and process each image
"""
if __name__ == "__main__":
    # for filename in os.listdir("./img/in/"):
    #     if filename.endswith(".jpg"):
    #         file_prefix = filename.split(".")[0]
    #         img = cv2.imread("./img/in/" + filename)
    #         face_detection(img, file_prefix + "-out")
    detect("./img/in/person-full.jpg")
