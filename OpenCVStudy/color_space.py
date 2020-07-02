import cv2

src = cv2.imread(r"1.jpg")

# dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# dst = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# print(dst.shape)
dst2 = cv2.convertScaleAbs(src, alpha=6, beta=1)

cv2.imshow("src show", src)
cv2.imshow("dst show", dst2)
cv2.waitKey(0)
