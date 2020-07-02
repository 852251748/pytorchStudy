# 近似形状
import cv2 as cv

img = cv.imread("26.jpg")

imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, imgThres = cv.threshold(imgGray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

contours, _ = cv.findContours(imgThres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(contours)
# approx = cv.approxPolyDP(contours[0], 50, True)

# img_contours = cv.drawContours(img, [approx], -1, (0, 0, 125), 3)
img_contours = cv.drawContours(img, contours, -1, (0, 0, 125), 3)
cv.imshow("img_contours", img_contours)
cv.waitKey(0)
