#!/usr/bin/env python3

import cv2
import sys
import numpy as np

FACE1 = 'haarcascades/haarcascade_profileface.xml'
FACE2 = 'haarcascades/haarcascade_frontalface_default.xml'
UBODY = 'haarcascades/haarcascade_upperbody.xml'

input_img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)

print('Original dimensions: ', input_img.shape)

scale = 1

width = int(input_img.shape[1] * scale)
height = int(input_img.shape[0] * scale)
dim = (width, height)

img = cv2.resize(input_img, dim, interpolation = cv2.INTER_AREA)

print('Resized dimensions: ', img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_classifier = cv2.CascadeClassifier(FACE2)

face = face_classifier.detectMultiScale(gray, 1.0485258, 6)

upperbody_classifier = cv2.CascadeClassifier(UBODY)

upperbody = upperbody_classifier.detectMultiScale(gray, 1.05, 5)

if face is ():
    print("No face found")
    sys.exit(0)
if upperbody is ():
    print("No upperbody detected")
    sys.exit(0)
x, y, w, h = face[0]
cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)
x, y, w, h = upperbody[0]
cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)
cv2.imwrite('output.jpg', img)

