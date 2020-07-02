import cv2
import matplotlib.pyplot as plt

img = cv2.imread('7.jpg', 0)
cv2.imshow("src", img)

his = cv2.calcHist([img], [0], None, [255], [0, 255])
plt.plot(his, label='his', color='r')
# plt.show()

dst = cv2.equalizeHist(img)
cv2.imshow("dst", dst)

cv2.imwrite("15.jpg", dst)

his = cv2.calcHist([dst], [0], None, [255], [0, 255])
plt.plot(his, label='his', color='b')
plt.show()