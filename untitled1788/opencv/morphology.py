# 图片形态学

import cv2 as cv

img1 = cv.imread("3.jpg")

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
imge = cv.erode(img1, kernel)
imgd = cv.dilate(img1, kernel)

sub = cv.subtract(imgd, imge)

cv.imshow("img1", img1)
cv.imshow("imge", imge)
cv.imshow("imgd", imgd)
cv.imshow("sub", sub)
cv.waitKey(0)
