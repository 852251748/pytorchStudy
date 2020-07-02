# 猪图片的集合变化
import cv2 as cv

img = cv.imread("src1.jpg")

imgGray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# retval, dst = cv.threshold(imgGray, 0, 255, cv.THRESH_TOZERO | cv.THRESH_OTSU)

dst1 = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 25, 10)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 3))
kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
# #去噪点
dst2 = cv.morphologyEx(dst1, cv.MORPH_OPEN, kernel)
cv.imshow("dst1", dst2)
cv.waitKey(0)
exit()
dst3 = cv.medianBlur(dst2, 5)
dst4 = cv.bilateralFilter(dst3, 9, 75, 75)
cv.imshow("dst3", dst3)
cv.imshow("dst4", dst4)
cv.waitKey(0)
exit()
dst3 = cv.morphologyEx(imgGray, cv.MORPH_GRADIENT, kernel1)

cv.imshow("dst4", dst4)
# cv.imshow("dst2", dst2)
cv.imshow("dst3", dst3)
cv.waitKey(0)
