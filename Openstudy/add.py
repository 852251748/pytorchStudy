import cv2
import numpy as np

a = np.uint8([100])
b = np.uint8([200])

c = cv2.add(a, b)
d = cv2.subtract(a, b)
print(c, d)

img1 = cv2.imread("1.jpg")
img2 = cv2.imread("6.jpg")

img = cv2.addWeighted(img1, 0.7, img2, 0.3,0)
cv2.imshow("img",img)
cv2.waitKey(0)