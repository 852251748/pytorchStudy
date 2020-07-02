import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

gray = cv.imread("../2.jpg",0)

# 普通二值化操作
_, thre0 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# 平均自适应阀值
thre1 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize=11, C=2)
# 高斯自适应阀值
thre2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=11, C=2)

title = ['Original imgae', 'Global Thresholding', 'AdaptiveThreshold Mean', 'AdaptiveThreshold Gaussian']

images = [gray, thre0, thre1, thre2]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])
plt.show()
