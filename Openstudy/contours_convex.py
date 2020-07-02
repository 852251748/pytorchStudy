#凸包检测
import cv2 as cv

img = cv.imread("26.jpg")

imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, imgThres = cv.threshold(imgGray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
contours,_ = cv.findContours(imgThres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

hull = cv.convexHull(contours[0])
print(cv.isContourConvex(hull), cv.isContourConvex(contours[0]))

img = cv.drawContours(img, [hull], -1, (0, 0, 255), 3)
cv.imshow("img", img)
cv.waitKey(0)
