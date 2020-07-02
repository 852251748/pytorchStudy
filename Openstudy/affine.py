# 仿射变换
import cv2
import numpy as np
import math

img = cv2.imread("1.jpg")
rows, cols, channals = img.shape
print()
# 沿X,Y轴平移50
# M = np.float32([[1, 0, 50], [0, 1, 50]])
# 将图片的高和宽都缩小一半
# M = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
# 将图片逆时针旋转10°
# M = np.float32([[math.cos(math.radians(10)), math.sin(math.radians(10)), 0],
#                 [-math.sin(math.radians(10)), math.cos(math.radians(10)), 0]])
# 将图片进行方向剪切
# M = np.float32([[1, 0, 0], [2, 1, 0]])

#自动得到需要的变换矩阵 第一个参数为旋转的中心，第二个参数为旋转的度数，第三个参数为缩放的比例因子
M = cv2.getRotationMatrix2D((cols/2,rows/2), 0, 2)

img = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow("img", img)
cv2.waitKey(0)
