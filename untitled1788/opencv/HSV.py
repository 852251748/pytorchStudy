import cv2 as cv
import numpy as np

img = cv.imread("15.jpg")
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lower = np.array([120, 102, 102])
high = np.array([123, 106, 107])
lowerb = np.array([130, 150, 130])
highb = np.array([140, 200, 200])
lowerb1 = np.array([100, 200, 100])
highb1 = np.array([200, 255, 200])

mask = cv.inRange(hsv, lower, high)
print(mask)
res = cv.bitwise_and(img, img, mask=mask)

cv.imshow("img", img)
cv.imshow("mask", hsv)
cv.waitKey(0)
