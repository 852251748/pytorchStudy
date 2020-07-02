import cv2
import numpy as np

img = cv2.imread("ts.jpg")
rows,cols,channal = img.shape
print(img.shape)
# point1 = np.float32([[25, 30], [179, 25], [12, 188], [189, 190]])
# point2 = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])


point1 = np.float32([[97, 61], [508, 71], [97, 503], [508, 503]])
point2 = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])

M = cv2.getPerspectiveTransform(point1, point2)

dst = cv2.warpPerspective(img, M, (cols,rows))

cv2.imshow("img",img)
cv2.imshow("dst",dst)
cv2.waitKey(0)
