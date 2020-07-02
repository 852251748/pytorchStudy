import cv2 as cv

img = cv.imread("1.jpg")

img = cv.GaussianBlur(img, (3, 3), 0)

img = cv.Canny(img, 50, 150)

cv.imshow("img", img)
cv.waitKey(0)
