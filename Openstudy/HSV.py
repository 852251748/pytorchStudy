import cv2
import numpy as np

src = cv2.imread(r"11.jpg")

dst = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

lower = np.array([100, 160, 100])
upper = np.array([250, 255, 200])

mask = cv2.inRange(dst, lower, upper)
res = cv2.bitwise_and(src, src, mask=mask)

cv2.imshow("mask", mask)
cv2.imshow("img", src)
cv2.imshow("res", res)
cv2.waitKey(0)
