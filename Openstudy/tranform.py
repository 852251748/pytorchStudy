import cv2

img = cv2.imread("1.jpg")

# img = cv2.transpose(img)
dst = cv2.flip(img, -1)
cv2.imshow("dst", dst)
cv2.imshow("img", img)
cv2.waitKey(0)
