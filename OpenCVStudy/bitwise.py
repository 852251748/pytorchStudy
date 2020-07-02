import cv2

img1 = cv2.imread('1.jpg')
img2 = cv2.imread('6.jpg')

# img = cv2.add(img1,img2)
dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
cv2.imshow("...",dst)
cv2.waitKey(0)

# rows, cols, channels = img2.shape
# roi = img1[0:rows, 0:cols]
#
# img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
# mask_inv = cv2.bitwise_not(mask)
#
#
# img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
# # cv2.imshow("img1_bg",img1_bg)
# # cv2.waitKey(0)
# # exit()
#
# img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
# cv2.imshow("img2_fg",img2_fg)
#
# dst = cv2.add(img1_bg, img2_fg)
# img1[0:rows, 0:cols] = dst
#
#
# cv2.imshow('res', img1)
# cv2.waitKey(0)