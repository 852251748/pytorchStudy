import cv2
import numpy as np
import matplotlib.pyplot as plt

roi = cv2.imread('part2.jpg')
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
target = cv2.imread('src1.jpg')
hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

# roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
roihist = cv2.calcHist([hsv], [0, 1 ], None, [180, 255], [0, 180, 0, 256])
# plt.plot(roihist, label='roihist', color='r')
# plt.show()
# 归一化 反投影
# cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)

disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dst = cv2.filter2D(dst, -1, disc)

ret, thresh = cv2.threshold(dst, 50, 255, 0)

cv2.imshow('thresh', thresh)
cv2.waitKey(0)
thresh = cv2.merge((thresh, thresh, thresh))

res = cv2.bitwise_and(target, thresh)
res = np.hstack((target, thresh, res))

cv2.imshow('img', res)
cv2.waitKey(0)
