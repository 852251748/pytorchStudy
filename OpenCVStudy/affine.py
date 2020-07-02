import cv2
import numpy as np

src = cv2.imread('1.jpg')

rows, cols, channel = src.shape

# M = np.float32([[1, 0, 50], [0, 1, 50]])
# M = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
# M = np.float32([[-0.5, 0, cols // 2], [0, 0.5, 0]])
# M = np.float32([[1, 0.5, 0], [0, 1, 0]])
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.7)
dst = cv2.warpAffine(src, M, (cols, rows))

cv2.imshow('src pic', src)
cv2.imshow('dst pic', dst)

cv2.waitKey(0)