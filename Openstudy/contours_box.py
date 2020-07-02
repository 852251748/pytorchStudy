import cv2 as cv
import numpy as np

img = cv.imread("16.jpg")

imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, imgthresh = cv.threshold(imgGray, 127, 255, 0)

contours, _ = cv.findContours(imgthresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# # 最小矩形
# rect = cv.minAreaRect(contours[0])
# point = cv.boxPoints(rect)
# box = np.int0(point)
# drawMinRect = cv.drawContours(img, [box], 0, (0, 0, 255), 1)
#
# # 最小外切圆
# (x, y), radius = cv.minEnclosingCircle(contours[0])
# center = (int(x), int(y))
# radius = int(radius)
# drawMinCircle = cv.circle(img, center, radius, (0, 255, 0), 1)

# 边界矩形
x, y, w, h = cv.boundingRect(contours[0])
drawRect = cv.rectangle(img, (x, y), (x + h, y + w), (255, 0, 0), 1)

cv.imshow("drawMinRect", img)
cv.waitKey(0)
