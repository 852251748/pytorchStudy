import cv2
import numpy as np

image = cv2.imread("29.jpg")
dst = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
circle = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 80, param1=40, param2=20, minRadius=20, maxRadius=300)
if not circle is None:
    circle = np.uint16(np.around(circle))
    for i in circle[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)

cv2.imshow("circle", image)
cv2.waitKey(0)