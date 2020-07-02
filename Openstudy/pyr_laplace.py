# 拉普拉斯金字塔
import cv2 as cv

img = cv.imread("12.jpg")
img_down = cv.pyrDown(img)
img_up = cv.pyrUp(img_down)
imglaplace = cv.subtract(img, img_up)
img_new = cv.add(imglaplace,img)
# 增加下对比度
imglaplace = cv.convertScaleAbs(imglaplace, alpha=6, beta=0)
cv.imshow("img", img)
cv.imshow("img_new",img_new)
cv.waitKey(0)
