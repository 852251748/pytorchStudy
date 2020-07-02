import cv2

img = cv2.imread('8.jpg', 0)
cv2.imshow("src", img)

dst1 = cv2.equalizeHist(img)
cv2.imshow("dst1", dst1)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
dst2 = clahe.apply(img)
cv2.imshow("dst2", dst2)

cv2.waitKey(0)