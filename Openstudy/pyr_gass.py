#高斯金字塔
import cv2 as cv

img = cv.imread("13.jpg")

for i in range(3):
    cv.imshow(f"img{i}",img)
    #下采样
    # img = cv.pyrDown(img)
    #上采样
    img = cv.pyrUp(img)
cv.waitKey(0)