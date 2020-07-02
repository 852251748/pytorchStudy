import cv2 as cv

img = cv.imread("src1.jpg")

imgGray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

ret, thresh = cv.threshold(imgGray, 127, 255, cv.THRESH_BINARY)

contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))
dst = cv.drawContours(img, contours, -1, (0, 255, 0), 2)
cv.imshow("dst", dst)
cv.waitKey(0)
exit()

img = cv.convertScaleAbs(img, alpha=1, beta=0)
cv.imshow("img1", img)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
cv.imshow("img2", img)
img = cv.Canny(img, 50, 150)

cv.imshow("img3", img)
cv.waitKey(0)
