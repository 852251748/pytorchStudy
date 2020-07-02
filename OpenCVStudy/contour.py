import cv2

img = cv2.imread('14.jpg')
# cv2.imshow("src", img)
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imggray, 127, 255, 0)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(len(contours[0]))
img_contour = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

cv2.imshow("img_contour", img_contour)
cv2.waitKey(0)