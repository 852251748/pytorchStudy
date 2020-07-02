import cv2

img1 = cv2.imread("1.jpg")
img2 = cv2.imread("6.jpg")
img3 = cv2.imread("9.jpg")

img3 = cv2.resize(img3, (300, 418))


# img = cv2.add(img1, img2)
# cv2.imshow("img", img)


rows, cols, channels = img2.shape
print(rows, cols)
roi = img1

img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi, img3)

cv2.imshow("img2", img2)
# cv2.imshow("mask_inv", mask_inv)
cv2.imshow("img1_bg", img1_bg)
cv2.waitKey(0)
