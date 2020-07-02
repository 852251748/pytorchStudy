import cv2
import numpy as np

img = cv2.imread("14.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red = np.array([78, 43, 46])
upper_red = np.array([99, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)
cv2.imshow('hsv', mask)
cv2.imshow('img', img)
cv2.waitKey(0)
