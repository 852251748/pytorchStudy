import cv2 as cv

img1 = cv.imread("../1.jpg")
img2 = cv.imread("6.jpg")

print(img2.shape, img1.shape)
addimg = cv.addWeighted(img1, 0.7, img2, 0.3, 0)
cv.bi
cv.imshow("addimg", addimg)
cv.waitKey(0)
