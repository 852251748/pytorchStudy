import cv2 as cv
import numpy as np

gray = cv.imread("../1.jpg", 0)

ret, thresh = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)
_, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
_, thresh2 = cv.threshold(gray, 0, 255, cv.THRESH_TOZERO | cv.THRESH_OTSU)

print(thresh, ret)
cv.imshow("img", thresh)
cv.imshow("img1", thresh1)
cv.imshow("img2", thresh2)
cv.waitKey(0)
