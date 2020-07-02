import cv2
import numpy as np

src = cv2.imread(r"riya.jpg")
src = cv2.resize(src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

dst = cv2.cvtColor(src, cv2.COLOR_BGR2HLS)
cv2.imshow("girl", src)
cv2.imshow("dst", dst)
cv2.waitKey(0)