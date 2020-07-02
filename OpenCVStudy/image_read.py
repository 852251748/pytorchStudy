# 图像读取
#
import cv2

img = cv2.imread(r"1.jpg",0)
cv2.imshow("pic show", img)
cv2.waitKey(0)
