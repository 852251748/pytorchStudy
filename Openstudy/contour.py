# 找轮廓
import cv2 as cv

img = cv.imread("14.jpg")

imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(imggray, 127, 255, cv.THRESH_BINARY)

contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(contours[0])

M = cv.moments(contours[0])
cx,cy = int(M['m10']/M['m00'])
img_contour = cv.drawContours(img, contours, -1, (0, 255, 0), 3)

cv.imshow("img1", img_contour)
cv.waitKey(0)
