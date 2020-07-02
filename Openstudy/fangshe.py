import cv2
import numpy as np
import math

a = np.array([[math.cos(math.radians(30)), math.sin(math.radians(30)), 0],
              [-math.sin(math.radians(30)), math.cos(math.radians(30)), 0], [0, 0, 1]])

print(a)

img = cv2.imread("1.jpg")
img[...:0] = img[...:0]*a
