# 滤波
import cv2 as cv
import numpy as np

img = cv.imread("2.jpg")

# 普通低通滤波
# kernel = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], np.float32)
# dst = cv.filter2D(img, -1, kernel)

# 以下低通滤波
# 均值滤波
# dst1 = cv.blur(img, (3, 3))
# 中值滤波
# dst = cv.medianBlur(img, 3)
# 高斯滤波
# dst = cv.GaussianBlur(img, (3, 3), 3)
# 双边滤波
# dst = cv.bilateralFilter(img, 9, 75, 75)


#
# 拉普拉斯算子  高通滤波
# dst = cv.Laplacian(img, -1)

# USM锐化 原图像素乘以两倍减去高斯模糊后的图 使得高频信号更为突出
# dst = cv.GaussianBlur(img, (3, 3), 2)
# dst = cv.addWeighted(img, 2, dst, -1, 0)

# 梯度算子
# 对x方向上求梯度
# dst = cv.Sobel(img, -1, 1, 0)
# 对y方向上求梯度
# dst = cv.Sobel(img,-1,0,1)
# 对x,y方向上求梯度
# dst = cv.Sobel(img, -1, 1, 1)
# Sobel改进版本
dst = cv.Scharr(img, -1, 1, 0)

cv.imshow("dst", dst)
cv.imshow("img", img)
cv.waitKey(0)
